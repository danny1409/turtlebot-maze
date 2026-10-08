-- The assignment's two tables, with provenance and raw measurements for
-- idempotent delivery, clock resets, and later association-radius evaluation.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS semantic_objects (
    object_id BIGSERIAL PRIMARY KEY,
    class_name TEXT NOT NULL,
    map_x DOUBLE PRECISION NOT NULL,
    map_y DOUBLE PRECISION NOT NULL,
    map_z DOUBLE PRECISION NOT NULL,
    observation_count INTEGER NOT NULL CHECK (observation_count > 0),
    mean_embedding vector(512) NOT NULL,
    first_seen_ns BIGINT NOT NULL,
    last_seen_ns BIGINT NOT NULL,
    embedding_model TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS semantic_objects_class_name_idx
    ON semantic_objects (class_name);

CREATE TABLE IF NOT EXISTS object_observations (
    observation_id BIGSERIAL PRIMARY KEY,
    object_id BIGINT NOT NULL REFERENCES semantic_objects (object_id) ON DELETE CASCADE,
    seen_at_ns BIGINT NOT NULL,
    map_x DOUBLE PRECISION NOT NULL,
    map_y DOUBLE PRECISION NOT NULL,
    map_yaw DOUBLE PRECISION NOT NULL,
    camera_x DOUBLE PRECISION NOT NULL,
    camera_y DOUBLE PRECISION NOT NULL,
    camera_z DOUBLE PRECISION NOT NULL,
    camera_qx DOUBLE PRECISION NOT NULL,
    camera_qy DOUBLE PRECISION NOT NULL,
    camera_qz DOUBLE PRECISION NOT NULL,
    camera_qw DOUBLE PRECISION NOT NULL,
    bearing_rad DOUBLE PRECISION NOT NULL,
    range_m DOUBLE PRECISION NOT NULL CHECK (range_m > 0),
    bbox_xmin INTEGER NOT NULL,
    bbox_ymin INTEGER NOT NULL,
    bbox_xmax INTEGER NOT NULL,
    bbox_ymax INTEGER NOT NULL,
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    embedding vector(512) NOT NULL,
    source_det_id TEXT NOT NULL UNIQUE,
    run_id TEXT NOT NULL,
    clock_epoch INTEGER NOT NULL,
    keyframe_id BIGINT NOT NULL,
    object_x DOUBLE PRECISION NOT NULL,
    object_y DOUBLE PRECISION NOT NULL,
    object_z DOUBLE PRECISION NOT NULL,
    association_radius_m DOUBLE PRECISION NOT NULL CHECK (association_radius_m > 0),
    CHECK (bbox_xmax > bbox_xmin AND bbox_ymax > bbox_ymin)
);
CREATE INDEX IF NOT EXISTS object_observations_object_id_idx
    ON object_observations (object_id);
