import { MongoDBAtlasVectorSearch } from "@langchain/mongodb";
import { OpenAIEmbeddings } from "@langchain/openai";
import { FastifyPluginAsync } from "fastify";
import fs from "fs/promises";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import z from "zod";
import { config } from "../../../config/env";
import { ZodTypeProvider } from "fastify-type-provider-zod";

const train: FastifyPluginAsync = async function (fastify, _opts) {
  // Define a Fastify plugin that handles a GET request to the root URL
  fastify.withTypeProvider<ZodTypeProvider>().route({
    method: "GET",
    url: "/",
    schema: {
      response: {
        200: z.object({
          message: z.string(),
        }),
      },
    },
    handler: async () => {
      try {
        // Access the MongoDB collection for embeddings
        const collection = fastify.mongo.client
          .db("research")
          .collection("embeddings");
        // Read files from the "./papers" directory
        const files = await fs.readdir("./papers");
        // Process each file in the directory
        for (const file of files) {
          // Load a PDF document
          const document = new PDFLoader(`./papers/${file}`);
          const load = await document.load();
          // Split the text of the document into chunks
          const textSplitter = new RecursiveCharacterTextSplitter({
            chunkSize: 100,
            chunkOverlap: 20,
          });
          const paper = await textSplitter.splitDocuments(load);
          // Store document embeddings in MongoDB Atlas
          await MongoDBAtlasVectorSearch.fromDocuments(
            paper,
            new OpenAIEmbeddings({
              apiKey: config.OPEN_API_KEY,
            }),
            {
              collection,
              indexName: "vector_index",
              textKey: "text",
              embeddingKey: "embeddings",
            }
          );
        }
        // Return a success message when embeddings are completed
        return { message: "embeddings completed!!" };
      } catch (error) {
        // Log any errors that occur during the process
        fastify.log.error(error);
      }
    },
  });
};

export default train;
