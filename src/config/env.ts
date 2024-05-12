import { envSchema } from "env-schema";
import { Static, Type } from "@sinclair/typebox";

const configSchema = Type.Object({
  OPEN_API_KEY: Type.String(),
  MONGO_URL: Type.String(),
});

type Config = Static<typeof configSchema>;

export const config = envSchema<Config>({
  schema: configSchema,
  dotenv: true,
  expandEnv: true,
});
