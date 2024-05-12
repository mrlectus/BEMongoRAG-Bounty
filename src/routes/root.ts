import { FastifyPluginAsync } from "fastify";
import { ZodTypeProvider } from "fastify-type-provider-zod";
import z from "zod";

const root: FastifyPluginAsync = async function (fastify, _opts) {
  fastify.withTypeProvider<ZodTypeProvider>().route({
    method: "GET",
    url: "/ping",
    schema: {
      response: {
        200: z.object({
          message: z.string(),
        }),
      },
    },
    handler: () => {
      return { message: "Pong" };
    },
  });
};

export default root;
