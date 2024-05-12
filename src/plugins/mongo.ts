import fp from "fastify-plugin";
import mongodb from "@fastify/mongodb";
import { config } from "../config/env";

export default fp(async (fastify) => {
  await fastify.register(mongodb, {
    forceClose: true,
    url: config.MONGO_URL,
  });
});
