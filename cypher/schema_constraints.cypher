// schema_constraints.cypher
// Run once after creating the Neo4j database (before first ingest).
// Open Neo4j Browser at localhost:7474 (or Aura Query panel) and paste,
// or run via `cypher-shell -f cypher/schema_constraints.cypher`.
//
// All VALID_NODE_LABELS from climber_network.vocab are listed here.

// --- Uniqueness constraints (guarantees MERGE idempotency) ---------------

CREATE CONSTRAINT athlete_id IF NOT EXISTS
  FOR (n:Athlete) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT event_id IF NOT EXISTS
  FOR (n:Event) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT round_id IF NOT EXISTS
  FOR (n:Round) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT performance_id IF NOT EXISTS
  FOR (n:Performance) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT discipline_id IF NOT EXISTS
  FOR (n:Discipline) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT rating_id IF NOT EXISTS
  FOR (n:Rating) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT venue_id IF NOT EXISTS
  FOR (n:Venue) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT city_id IF NOT EXISTS
  FOR (n:City) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT country_id IF NOT EXISTS
  FOR (n:Country) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT timezone_id IF NOT EXISTS
  FOR (n:TimeZone) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT travelleg_id IF NOT EXISTS
  FOR (n:TravelLeg) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT resty_id IF NOT EXISTS
  FOR (n:RestednessState) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT trainingsignal_id IF NOT EXISTS
  FOR (n:TrainingSignal) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT injuryevent_id IF NOT EXISTS
  FOR (n:InjuryEvent) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT trainingcamp_id IF NOT EXISTS
  FOR (n:TrainingCamp) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT source_id IF NOT EXISTS
  FOR (n:Source) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT document_id IF NOT EXISTS
  FOR (n:Document) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT extractionrun_id IF NOT EXISTS
  FOR (n:ExtractionRun) REQUIRE n.id IS UNIQUE;

// --- Additional indexes --------------------------------------------------

// Shared :Entity label index — every node also carries the :Entity label
// (stamped by GraphClient.merge_node[s]). Relationship MERGEs match endpoints
// by id WITHOUT knowing their specific label, so they rely on THIS index;
// a per-label id index can't serve an unlabeled / cross-label id lookup, which
// would otherwise force a full node scan per row (catastrophic at 30k+ edges).
CREATE INDEX entity_id IF NOT EXISTS
  FOR (n:Entity) ON (n.id);

// Full-text / name searches on athletes
CREATE INDEX athlete_name IF NOT EXISTS
  FOR (n:Athlete) ON (n.name);

// Geospatial queries on venue locations (neo4j POINT type)
CREATE POINT INDEX venue_loc IF NOT EXISTS
  FOR (n:Venue) ON (n.location);

// --- Verify ---------------------------------------------------------------

SHOW CONSTRAINTS;
