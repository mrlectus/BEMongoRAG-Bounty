import { MongoDBAtlasVectorSearch } from "@langchain/mongodb";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { FastifyPluginAsync } from "fastify";
import { ZodTypeProvider } from "fastify-type-provider-zod";
import z from "zod";
import { config } from "../../../config/env";
import { PromptTemplate } from "@langchain/core/prompts";
import {
  RunnablePassthrough,
  RunnableSequence,
} from "@langchain/core/runnables";
import { formatDocumentsAsString } from "langchain/util/document";
import { StringOutputParser } from "@langchain/core/output_parsers";

const chat: FastifyPluginAsync = async function (fastify, _opts) {
  fastify.withTypeProvider<ZodTypeProvider>().route({
    method: "POST",
    url: "/",
    schema: {
      response: {
        200: z.object({
          answer: z.string(),
        }),
      },
      body: z.object({
        message: z.string(),
      }),
    },
    handler: async (request) => {
      const { message } = request.body;
      try {
        const collection = fastify.mongo.client
          .db("research")
          .collection("embeddings");
        const vectorStore = new MongoDBAtlasVectorSearch(
          new OpenAIEmbeddings({
            modelName: "text-embedding-ada-002",
            stripNewLines: true,
            apiKey: config.OPEN_API_KEY,
          }),
          {
            collection,
            indexName: "vector_index",
            textKey: "text",
            embeddingKey: "embeddings",
          }
        );

        const retriver = vectorStore.asRetriever({
          searchType: "mmr",
          searchKwargs: {
            fetchK: 50,
            lambda: 0.1,
          },
        });

        const prompt = PromptTemplate.fromTemplate(`
        You are a Research Assistant tasked with providing detailed summaries of academic articles related to the given context. Please provide a summary of the most relevant articles, including the article title, authors, and year of publication if available. Format your response in markdown. And if you do not have an answer say you don't.
          Context: {context}
        Question: {question}`);

        const model = new ChatOpenAI({
          apiKey: config.OPEN_API_KEY,
        });

        const chain = RunnableSequence.from([
          {
            context: retriver.pipe(formatDocumentsAsString),
            question: new RunnablePassthrough(),
          },
          prompt,
          model,
          new StringOutputParser(),
        ]);

        const retriverOutput = await chain.invoke(message);
        return { answer: retriverOutput };
      } catch (error) {
        fastify.log.error(error);
      }
    },
  });
};

export default chat;
