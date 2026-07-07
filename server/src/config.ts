import "dotenv/config";

export const config = {
  port: parseInt(process.env.PORT || "3000", 10),
  dbPath: process.env.DB_PATH || "./data/cam.db",
  jwtSecret: process.env.JWT_SECRET || "dev-secret-change-in-production",
  mqttUrl: process.env.MQTT_URL || "mqtt://localhost:1883",
  minioEndpoint: process.env.MINIO_ENDPOINT || "localhost:9000",
  minioAccessKey: process.env.MINIO_ACCESS_KEY || "minioadmin",
  minioSecretKey: process.env.MINIO_SECRET_KEY || "minioadmin",
  cloudApiUrl: process.env.CLOUD_API_URL,
  cloudApiKey: process.env.CLOUD_API_KEY,
  tenantId: process.env.TENANT_ID || "default",
  nodeEnv: process.env.NODE_ENV || "development",
};
