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
        const collection = fastify.mongo.client
          .db("research")
          .collection("embeddings");
        const files = await fs.readdir("./papers");
        for (const file of files) {
          const document = new PDFLoader(`./papers/${file}`);
          const load = await document.load();
          const textSplitter = new RecursiveCharacterTextSplitter({
            chunkSize: 100,
            chunkOverlap: 20,
          });
          const paper = await textSplitter.splitDocuments(load);
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
        return { message: "embeddings completed!!" };
      } catch (error) {
        fastify.log.error(error);
      }
    },
  });
};

export default train;
