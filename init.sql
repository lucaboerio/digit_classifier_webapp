CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    digit INTEGER NOT NULL,
    probability FLOAT NOT NULL,
    timestamp TIMESTAMP NOT NULL
);