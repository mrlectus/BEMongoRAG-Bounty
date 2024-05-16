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
import { sleep } from "langchain/util/time";

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
      const { message: question } = request.body;
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
        const retriever = vectorStore.asRetriever({
          searchType: "similarity",
        });

        // Create a prompt template for generating the query
        const prompt = PromptTemplate.fromTemplate(`
        You are an AI Research assistant that provides abstract, title, year \
        of publication and authors of articles or papers in proper markdown format. \
        if you have no answer, please respond with "No refrence to data you are looking for". \
        please provide summary of the abstract \
        Answer the question based on the following context: \
        Context: {context} \
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
            context: retriever.pipe(formatDocumentsAsString),
            question: new RunnablePassthrough(),
          },
          prompt,
          model,
          new StringOutputParser(),
        ]);

        await sleep(3000);
        // Invoke the pipeline with the user's message
        const retriverOutput = await chain.invoke(question);
        const retrievedResults = await retriever._getRelevantDocuments(
          question
        );
        const documents = retrievedResults.map((document) => ({
          pageContent: document.pageContent,
          pageNumber: document.metadata.loc.pageNumber,
        }));

        fastify.log.info(JSON.stringify(documents));

        return { answer: retriverOutput };
      } catch (error) {
        fastify.log.error(error);
      }
    },
  });
};
export default chat;
