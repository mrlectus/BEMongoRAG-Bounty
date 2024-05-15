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
        // Connect to the MongoDB database and collection
        const collection = fastify.mongo.client
          .db("research")
          .collection("embeddings");

        // Create a vector store for searching embeddings
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

        // Create a retriever using the vector store
        const retriver = vectorStore.asRetriever({
          searchType: "mmr",
          searchKwargs: {
            fetchK: 50,
            lambda: 0.1,
          },
        });

        // Create a prompt template for generating the query
        const prompt = PromptTemplate.fromTemplate(`
        You are an AI Research Assistant tasked with providing detailed summaries of academic articles including the summary, name, year of publication and authors in markdown format. If you can't find anything Say "I can't find this information you are looking for".
        Context: {context}
        Question: {question}`);

        // Create a chat model for generating the answer
        const model = new ChatOpenAI({
          modelName: "gpt-3.5-turbo",
          apiKey: config.OPEN_API_KEY,
          temperature: 0,
        });

        // Create a runnable sequence for executing the pipeline
        const chain = RunnableSequence.from([
          {
            context: retriver.pipe(formatDocumentsAsString),
            question: new RunnablePassthrough(),
          },
          prompt,
          model,
          new StringOutputParser(),
        ]);

        // Invoke the pipeline with the user's message
        const retriverOutput = await chain.invoke(message);
        return { answer: retriverOutput };
      } catch (error) {
        fastify.log.error(error);
      }
    },
  });
};
export default chat;
