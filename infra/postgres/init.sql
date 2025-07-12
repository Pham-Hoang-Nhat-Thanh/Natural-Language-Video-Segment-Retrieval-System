-- Create database if not exists
CREATE DATABASE IF NOT EXISTS video_retrieval;

-- Create user if not exists
DO $$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'video_user') THEN
      CREATE USER video_user WITH PASSWORD 'video_password';
   END IF;
END
$$;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE video_retrieval TO video_user;

-- Switch to video_retrieval database
\c video_retrieval;

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create tables will be handled by the application
