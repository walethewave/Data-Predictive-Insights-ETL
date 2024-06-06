CREATE TABLE IF NOT EXISTS Athlete (
  athlete_id SERIAL PRIMARY KEY,
  Name TEXT,
  Sex TEXT,
  Age INTEGER,
  AgeClass TEXT,
  BodyweightKg REAL,
  Country TEXT,
  Tested TEXT
);

CREATE TABLE IF NOT EXISTS Event (
  event_id SERIAL PRIMARY KEY,
  Event TEXT
);

CREATE TABLE IF NOT EXISTS Equipment (
  equipment_id SERIAL PRIMARY KEY,
  Equipment TEXT
);

CREATE TABLE IF NOT EXISTS Division (
  division_id SERIAL PRIMARY KEY,
  Division TEXT
);

CREATE TABLE IF NOT EXISTS WeightClass (
  weightclass_id SERIAL PRIMARY KEY,
  WeightClassKg TEXT
);

CREATE TABLE IF NOT EXISTS Lift (
  lift_id SERIAL PRIMARY KEY,
  athlete_id INTEGER REFERENCES Athlete(athlete_id),
  event_id INTEGER REFERENCES Event(event_id),
  equipment_id INTEGER REFERENCES Equipment(equipment_id),
  division_id INTEGER REFERENCES Division(division_id),
  weightclass_id INTEGER REFERENCES WeightClass(weightclass_id),
  Squat1Kg REAL,
  Squat2Kg REAL,
  Squat3Kg REAL,
  Squat4Kg REAL,
  Best3SquatKg REAL,
  Bench1Kg REAL,
  Bench2Kg REAL,
  Bench3Kg REAL,
  Bench4Kg REAL,
  Best3BenchKg REAL,
  Deadlift1Kg REAL,
  Deadlift2Kg REAL,
  Deadlift3Kg REAL,
  Deadlift4Kg REAL,
  Best3DeadliftKg REAL,
  TotalKg REAL
);

CREATE TABLE IF NOT EXISTS Meet (
  meet_id SERIAL PRIMARY KEY,
  MeetCountry TEXT,
  MeetState TEXT,
  MeetName TEXT,
  Date DATE
);

CREATE TABLE IF NOT EXISTS Performance (
  performance_id SERIAL PRIMARY KEY,
  athlete_id INTEGER REFERENCES Athlete(athlete_id),
  meet_id INTEGER REFERENCES Meet(meet_id),
  Place INTEGER,
  Wilks REAL,
  McCulloch REAL,
  Glossbrenner REAL,
  IPFPoints REAL,
  Tested TEXT
);
