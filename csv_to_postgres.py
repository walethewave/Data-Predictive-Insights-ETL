import pandas as pd
import psycopg2
import config
import re
from tqdm import tqdm
 
# Function to establish a connection to the PostgreSQL database
def connect_to_db(dbname):
    try:
        conn = psycopg2.connect(
            dbname=dbname,
            user=config.user,
            password=config.password,
            host=config.host,
            port=config.port
        )
        print("Connected to the database")
        return conn
    except psycopg2.Error as e:
        print("Error connecting to the database:", e)
        return None
 
# Function to create tables if they don't exist
def create_tables(conn, dbname):
    try:
        cur = conn.cursor()
 
        # Create Athlete table
        cur.execute("""
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
        """)
 
        # Create Event table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS Event (
                event_id SERIAL PRIMARY KEY,
                Event TEXT
            );
        """)
 
        # Create Equipment table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS Equipment (
                equipment_id SERIAL PRIMARY KEY,
                Equipment TEXT
            );
        """)
 
        # Create Division table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS Division (
                division_id SERIAL PRIMARY KEY,
                Division TEXT
            );
        """)
 
        # Create WeightClass table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS WeightClass (
                weightclass_id SERIAL PRIMARY KEY,
                WeightClassKg TEXT
            );
        """)
 
        # Create Lift table
        cur.execute("""
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
        """)
 
        # Create Meet table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS Meet (
                meet_id SERIAL PRIMARY KEY,
                MeetCountry TEXT,
                MeetState TEXT,
                MeetName TEXT,
                Date DATE
            );
        """)
 
        if dbname == "openpowerlifting":
            # Create Performance table for openpowerlifting database
            cur.execute("""
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
            """)
        elif dbname == "openpowerlifting-2024-01-06-4c732975":
            # Create Performance table for openpowerlifting_2024 database
            cur.execute("""
                CREATE TABLE IF NOT EXISTS Performance (
                    performance_id SERIAL PRIMARY KEY,
                    athlete_id INTEGER REFERENCES Athlete(athlete_id),
                    meet_id INTEGER REFERENCES Meet(meet_id),
                    Place INTEGER,
                    Wilks REAL,
                    Glossbrenner REAL,
                    Tested TEXT
                );
            """)
        else:
            print("Invalid database name.")
            return
 
        conn.commit()
        print("Tables created successfully")
    except psycopg2.Error as e:
        print("Error creating tables:", e)
 
# Function to insert data from openpowerlifting.csv into PostgreSQL tables
def insert_data_openpowerlifting(conn):
    try:
        cur = conn.cursor()
 
        # Read CSV file
        df = pd.read_csv("./Csv_data/openpowerlifting.csv", low_memory=False)
 
        # Define the preprocess_weight function
        def preprocess_weight(value):
            # Check if the value is a float
            if isinstance(value, float):
                return str(value)  # Convert float to string
            else:
                # Define the pattern to remove non-digit characters
                pattern = r'\D'
                # Remove non-digit characters from the value
                cleaned_value = re.sub(pattern, '', str(value))
                return cleaned_value
 
        # Apply the preprocess_weight function to the WeightClassKg column and store as strings
        df['WeightClassKg'] = df['WeightClassKg'].apply(preprocess_weight).astype(str)
 
        # Update Country and Tested columns if empty
        df['Testaed'] = df['Tested'].fillna('No')
 
        # Update MeetState column if empty
        df['MeetState'] = df['MeetState'].apply(lambda x: None if pd.isna(x) or x.strip() == '' else x)
 
        # Insert data into Athlete table
        df_athlete = df[['Name', 'Sex', 'Age', 'AgeClass', 'BodyweightKg', 'Country', 'Tested']]
        for idx, row in tqdm(df_athlete.iterrows(), total=len(df_athlete), desc="Inserting into Athlete"):
            try:
                # Convert NaN values in Age column to 0
                if pd.isna(row['Age']):
                    row['Age'] = 0
 
                # Set Country to None if empty
                if pd.isna(row['Country']) or row['Country'].strip() == '':
                    country = None
                else:
                    country = row['Country']
 
                cur.execute("""
                                INSERT INTO Athlete (Name, Sex, Age, AgeClass, BodyweightKg, Country, Tested)
                                VALUES (%s, %s, %s, %s, %s, %s, %s)
                                RETURNING athlete_id;
                            """, (row['Name'], row['Sex'], row['Age'], row['AgeClass'], row['BodyweightKg'], country, row['Tested']))
            except psycopg2.Error as e:
                print(f"Error inserting row {idx} into Athlete table: {e}")
                print("Row data:", row)
                # Save the problematic row to a separate table for inspection
                cur.execute("""
                    INSERT INTO ProblematicRows (row_index, error_message, row_data)
                    VALUES (%s, %s, %s);
                """, (idx, str(e), str(row)))
 
                athlete_id = cur.fetchone()[0]
            except psycopg2.Error as e:
                print(f"Error inserting row {idx} into Athlete table: {e}")
                print("Row data:", row)
                # Save the problematic row to a separate table for inspection
                cur.execute("""
                    INSERT INTO ProblematicRows (row_index, error_message, row_data)
                    VALUES (%s, %s, %s);
                """, (idx, str(e), str(row)))
                continue
 
        # Insert data into Event table
        df_event = df[['Event']].drop_duplicates()
        for _, row in tqdm(df_event.iterrows(), total=len(df_event), desc="Inserting into Event"):
            cur.execute("""
                INSERT INTO Event (Event)
                VALUES (%s)
                ON CONFLICT DO NOTHING;
            """, tuple(row))
 
        # Insert data into Equipment table
        df_equipment = df[['Equipment']].drop_duplicates()
        for _, row in tqdm(df_equipment.iterrows(), total=len(df_equipment), desc="Inserting into Equipment"):
            cur.execute("""
                INSERT INTO Equipment (Equipment)
                VALUES (%s)
                ON CONFLICT DO NOTHING;
            """, tuple(row))
 
        # Insert data into Division table
        df_division = df[['Division']].drop_duplicates()
        for _, row in tqdm(df_division.iterrows(), total=len(df_division), desc="Inserting into Division"):
            cur.execute("""
                INSERT INTO Division (Division)
                VALUES (%s)
                ON CONFLICT DO NOTHING;
            """, tuple(row))
 
        # Insert data into WeightClass table
        df_weightclass = df[['WeightClassKg']].drop_duplicates()
        for _, row in tqdm(df_weightclass.iterrows(), total=len(df_weightclass), desc="Inserting into WeightClass"):
            try:
                cur.execute("""
                    INSERT INTO WeightClass (WeightClassKg)
                    VALUES (%s)
                    ON CONFLICT DO NOTHING;
                """, (row['WeightClassKg'],))
            except psycopg2.Error as e:
                print(f"Error inserting row {_} into WeightClass table: {e}")
                print("Row data:", row)
                # Save the problematic row to a separate table for inspection
                cur.execute("""
                    INSERT INTO ProblematicRows (row_index, error_message, row_data)
                    VALUES (%s, %s, %s);
                """, (_, str(e), str(row)))
 
        # Insert data into Lift table
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Inserting into Lift"):
            try:
                # Replace empty values with None for Squat and Bench columns
                for col in ['Squat1Kg', 'Squat2Kg', 'Squat3Kg', 'Squat4Kg', 'Best3SquatKg',
                            'Bench1Kg', 'Bench2Kg', 'Bench3Kg', 'Bench4Kg', 'Best3BenchKg',
                            'Deadlift1Kg', 'Deadlift2Kg', 'Deadlift3Kg', 'Deadlift4Kg', 'Best3DeadliftKg',
                            'TotalKg']:
                    if pd.isna(row[col]):
                        row[col] = None
 
                # Handle NaN values in Division column
                if pd.isna(row['Division']):
                    division = None
                else:
                    division = row['Division']
 
                cur.execute("""
                    INSERT INTO Lift (athlete_id, event_id, equipment_id, division_id, weightclass_id,
                                    Squat1Kg, Squat2Kg, Squat3Kg, Squat4Kg, Best3SquatKg,
                                    Bench1Kg, Bench2Kg, Bench3Kg, Bench4Kg, Best3BenchKg,
                                    Deadlift1Kg, Deadlift2Kg, Deadlift3Kg, Deadlift4Kg, Best3DeadliftKg,
                                    TotalKg)
                    VALUES (
                        (SELECT athlete_id FROM Athlete WHERE Name = %s LIMIT 1),
                        (SELECT event_id FROM Event WHERE Event = %s LIMIT 1),
                        (SELECT equipment_id FROM Equipment WHERE Equipment = %s LIMIT 1),
                        (SELECT division_id FROM Division WHERE Division = %s LIMIT 1),
                        (SELECT weightclass_id FROM WeightClass WHERE WeightClassKg = %s LIMIT 1),
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        %s)
                """, (row['Name'], row['Event'], row['Equipment'], division, row['WeightClassKg'],
                    row['Squat1Kg'], row['Squat2Kg'], row['Squat3Kg'], row['Squat4Kg'], row['Best3SquatKg'],
                    row['Bench1Kg'], row['Bench2Kg'], row['Bench3Kg'], row['Bench4Kg'], row['Best3BenchKg'],
                    row['Deadlift1Kg'], row['Deadlift2Kg'], row['Deadlift3Kg'], row['Deadlift4Kg'], row['Best3DeadliftKg'],
                    row['TotalKg']))
            except psycopg2.Error as e:
                print(f"Error inserting row {idx} into Lift table: {e}")
                print("Row data:", row)
                # Save the problematic row to a separate table for inspection
                cur.execute("""
                    INSERT INTO ProblematicRows (row_index, error_message, row_data)
                    VALUES (%s, %s, %s);
                """, (idx, str(e), str(row)))
 
        # Insert data into Meet table
        df_meet = df[['MeetCountry', 'MeetState', 'MeetName', 'Date']].drop_duplicates()
        for _, row in tqdm(df_meet.iterrows(), total=len(df_meet), desc="Inserting into Meet"):
            cur.execute("""
                INSERT INTO Meet (MeetCountry, MeetState, MeetName, Date)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT DO NOTHING;
            """, tuple(row))
 
       # Insert data into Performance table
        df_performance = df[['Name', 'MeetCountry', 'MeetState', 'MeetName', 'Date', 'Place', 'Wilks', 'McCulloch', 'Glossbrenner', 'IPFPoints', 'Tested']]
        for idx, row in tqdm(df_performance.iterrows(), total=len(df_performance), desc="Inserting into Performance"):
            # Replace "DQ" with None in the Place column
        # Convert string 'DQ' to None in the Place column
            if isinstance(row['Place'], str):
                row['Place'] = None
 
            # Replace NaN values with None
            row['Wilks'] = None if pd.isna(row['Wilks']) else row['Wilks']
            row['McCulloch'] = None if pd.isna(row['McCulloch']) else row['McCulloch']
            row['Glossbrenner'] = None if pd.isna(row['Glossbrenner']) else row['Glossbrenner']
            row['IPFPoints'] = None if pd.isna(row['IPFPoints']) else row['IPFPoints']
            try:
                cur.execute("""
                    INSERT INTO Performance (athlete_id, meet_id, Place, Wilks, McCulloch, Glossbrenner, IPFPoints)
                    VALUES (
                        (SELECT athlete_id FROM Athlete WHERE Name = %s LIMIT 1),
                        (SELECT meet_id FROM Meet WHERE MeetCountry = %s AND MeetState = %s AND MeetName = %s AND Date = %s LIMIT 1),
                        %s, %s, %s, %s, %s)
                """, (row['Name'], row['MeetCountry'], row['MeetState'], row['MeetName'], row['Date'], row['Place'], row['Wilks'], row['McCulloch'], row['Glossbrenner'], row['IPFPoints']))
            except psycopg2.Error as e:
                print(f"Error inserting row {idx} into Performance table: {e}")
                print("Row data:", row)
                # Save the problematic row to a separate table for inspection
                cur.execute("""
                    INSERT INTO ProblematicRows (row_index, error_message, row_data)
                    VALUES (%s, %s, %s);
                """, (idx, str(e), str(row)))
 
        conn.commit()
        print("Data insertion from openpowerlifting.csv completed successfully")
    except psycopg2.Error as e:
        print("Error inserting data from openpowerlifting.csv:", e)
 
    # Pause execution for debugging
    input("Press any key to continue...")
 
# Function to insert data from openpowerlifting-2024-01-06-4c732975.csv into PostgreSQL tables
def insert_data_openpowerlifting_2024(conn):
    try:
        cur = conn.cursor()
 
        # Read CSV file
        df = pd.read_csv(".Csv_data/openpowerlifting-2024-01-06-4c732975.csv", low_memory=False)
 
        # Define the preprocess_weight function
        def preprocess_weight(value):
            # Check if the value is a float
            if isinstance(value, float):
                return str(value)  # Convert float to string
            else:
                # Define the pattern to remove non-digit characters
                pattern = r'\D'
                # Remove non-digit characters from the value
                cleaned_value = re.sub(pattern, '', str(value))
                return cleaned_value
 
        # Apply the preprocess_weight function to the WeightClassKg column and store as strings
        df['WeightClassKg'] = df['WeightClassKg'].apply(preprocess_weight).astype(str)
 
        # Update Country and Tested columns if empty
        df['Tested'] = df['Tested'].fillna('No')
 
        # Update MeetState column if empty
        df['MeetState'] = df['MeetState'].apply(lambda x: None if pd.isna(x) or x.strip() == '' else x)
 
        # Insert data into Athlete table
        df_athlete = df[['Name', 'Sex', 'Age', 'AgeClass', 'BodyweightKg', 'Country', 'Tested']]
        for idx, row in tqdm(df_athlete.iterrows(), total=len(df_athlete), desc="Inserting into Athlete"):
            try:
                # Convert NaN values in Age column to 0
                if pd.isna(row['Age']):
                    row['Age'] = 0
 
                # Set Country to None if empty
                if pd.isna(row['Country']) or row['Country'].strip() == '':
                    country = None
                else:
                    country = row['Country']
 
                cur.execute("""
                                INSERT INTO Athlete (Name, Sex, Age, AgeClass, BodyweightKg, Country, Tested)
                                VALUES (%s, %s, %s, %s, %s, %s, %s)
                                RETURNING athlete_id;
                            """, (row['Name'], row['Sex'], row['Age'], row['AgeClass'], row['BodyweightKg'], country, row['Tested']))
            except psycopg2.Error as e:
                print(f"Error inserting row {idx} into Athlete table: {e}")
                print("Row data:", row)
                # Save the problematic row to a separate table for inspection
                cur.execute("""
                    INSERT INTO ProblematicRows (row_index, error_message, row_data)
                    VALUES (%s, %s, %s);
                """, (idx, str(e), str(row)))
 
                athlete_id = cur.fetchone()[0]
            except psycopg2.Error as e:
                print(f"Error inserting row {idx} into Athlete table: {e}")
                print("Row data:", row)
                # Save the problematic row to a separate table for inspection
                cur.execute("""
                    INSERT INTO ProblematicRows (row_index, error_message, row_data)
                    VALUES (%s, %s, %s);
                """, (idx, str(e), str(row)))
                continue
 
        # Insert data into Event table
        df_event = df[['Event']].drop_duplicates()
        for _, row in tqdm(df_event.iterrows(), total=len(df_event), desc="Inserting into Event"):
            cur.execute("""
                INSERT INTO Event (Event)
                VALUES (%s)
                ON CONFLICT DO NOTHING;
            """, tuple(row))
 
        # Insert data into Equipment table
        df_equipment = df[['Equipment']].drop_duplicates()
        for _, row in tqdm(df_equipment.iterrows(), total=len(df_equipment), desc="Inserting into Equipment"):
            cur.execute("""
                INSERT INTO Equipment (Equipment)
                VALUES (%s)
                ON CONFLICT DO NOTHING;
            """, tuple(row))
 
        # Insert data into Division table
        df_division = df[['Division']].drop_duplicates()
        for _, row in tqdm(df_division.iterrows(), total=len(df_division), desc="Inserting into Division"):
            cur.execute("""
                INSERT INTO Division (Division)
                VALUES (%s)
                ON CONFLICT DO NOTHING;
            """, tuple(row))
 
        # Insert data into WeightClass table
        df_weightclass = df[['WeightClassKg']].drop_duplicates()
        for _, row in tqdm(df_weightclass.iterrows(), total=len(df_weightclass), desc="Inserting into WeightClass"):
            try:
                cur.execute("""
                    INSERT INTO WeightClass (WeightClassKg)
                    VALUES (%s)
                    ON CONFLICT DO NOTHING;
                """, (row['WeightClassKg'],))
            except psycopg2.Error as e:
                print(f"Error inserting row {_} into WeightClass table: {e}")
                print("Row data:", row)
                # Save the problematic row to a separate table for inspection
                cur.execute("""
                    INSERT INTO ProblematicRows (row_index, error_message, row_data)
                    VALUES (%s, %s, %s);
                """, (_, str(e), str(row)))
 
        # Insert data into Lift table
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Inserting into Lift"):
            try:
                # Replace empty values with None for Squat and Bench columns
                for col in ['Squat1Kg', 'Squat2Kg', 'Squat3Kg', 'Squat4Kg', 'Best3SquatKg',
                            'Bench1Kg', 'Bench2Kg', 'Bench3Kg', 'Bench4Kg', 'Best3BenchKg',
                            'Deadlift1Kg', 'Deadlift2Kg', 'Deadlift3Kg', 'Deadlift4Kg', 'Best3DeadliftKg',
                            'TotalKg']:
                    if pd.isna(row[col]):
                        row[col] = None
 
                # Handle NaN values in Division column
                if pd.isna(row['Division']):
                    division = None
                else:
                    division = row['Division']
 
                cur.execute("""
                    INSERT INTO Lift (athlete_id, event_id, equipment_id, division_id, weightclass_id,
                                    Squat1Kg, Squat2Kg, Squat3Kg, Squat4Kg, Best3SquatKg,
                                    Bench1Kg, Bench2Kg, Bench3Kg, Bench4Kg, Best3BenchKg,
                                    Deadlift1Kg, Deadlift2Kg, Deadlift3Kg, Deadlift4Kg, Best3DeadliftKg,
                                    TotalKg)
                    VALUES (
                        (SELECT athlete_id FROM Athlete WHERE Name = %s LIMIT 1),
                        (SELECT event_id FROM Event WHERE Event = %s LIMIT 1),
                        (SELECT equipment_id FROM Equipment WHERE Equipment = %s LIMIT 1),
                        (SELECT division_id FROM Division WHERE Division = %s LIMIT 1),
                        (SELECT weightclass_id FROM WeightClass WHERE WeightClassKg = %s LIMIT 1),
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        %s)
                """, (row['Name'], row['Event'], row['Equipment'], division, row['WeightClassKg'],
                    row['Squat1Kg'], row['Squat2Kg'], row['Squat3Kg'], row['Squat4Kg'], row['Best3SquatKg'],
                    row['Bench1Kg'], row['Bench2Kg'], row['Bench3Kg'], row['Bench4Kg'], row['Best3BenchKg'],
                    row['Deadlift1Kg'], row['Deadlift2Kg'], row['Deadlift3Kg'], row['Deadlift4Kg'], row['Best3DeadliftKg'],
                    row['TotalKg']))
            except psycopg2.Error as e:
                print(f"Error inserting row {idx} into Lift table: {e}")
                print("Row data:", row)
                # Save the problematic row to a separate table for inspection
                cur.execute("""
                    INSERT INTO ProblematicRows (row_index, error_message, row_data)
                    VALUES (%s, %s, %s);
                """, (idx, str(e), str(row)))
 
        # Insert data into Meet table
        df_meet = df[['MeetCountry', 'MeetState', 'MeetName', 'Date']].drop_duplicates()
        for _, row in tqdm(df_meet.iterrows(), total=len(df_meet), desc="Inserting into Meet"):
            cur.execute("""
                INSERT INTO Meet (MeetCountry, MeetState, MeetName, Date)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT DO NOTHING;
            """, tuple(row))
 
       # Insert data into Performance table
        df_performance = df[['Name', 'MeetCountry', 'MeetState', 'MeetName', 'Date', 'Place', 'Wilks', 'Glossbrenner', 'Tested']]
        for idx, row in tqdm(df_performance.iterrows(), total=len(df_performance), desc="Inserting into Performance"):
            # Replace "DQ" with None in the Place column
            if isinstance(row['Place'], str):
                row['Place'] = None

            # Replace NaN values with None
            row['Wilks'] = None if pd.isna(row['Wilks']) else row['Wilks']
            row['Glossbrenner'] = None if pd.isna(row['Glossbrenner']) else row['Glossbrenner']
            try:
                cur.execute("""
                    INSERT INTO Performance (athlete_id, meet_id, Place, Wilks, Glossbrenner, Tested)
                    VALUES (
                        (SELECT athlete_id FROM Athlete WHERE Name = %s LIMIT 1),
                        (SELECT meet_id FROM Meet WHERE MeetCountry = %s AND MeetState = %s AND MeetName = %s AND Date = %s LIMIT 1),
                        %s, %s, %s, %s)
                """, (row['Name'], row['MeetCountry'], row['MeetState'], row['MeetName'], row['Date'], row['Place'], row['Wilks'], row['Glossbrenner'], row['Tested']))
            except psycopg2.Error as e:
                print(f"Error inserting row {idx} into Performance table: {e}")
                print("Row data:", row)
                # Save the problematic row to a separate table for inspection
                cur.execute("""
                    INSERT INTO ProblematicRows (row_index, error_message, row_data)
                    VALUES (%s, %s, %s);
                """, (idx, str(e), str(row)))
 
        conn.commit()
        print("Data insertion from openpowerlifting-2024-01-06-4c732975.csv completed successfully")
    except psycopg2.Error as e:
        print("Error inserting data from openpowerlifting-2024-01-06-4c732975.csv:", e)
 
    # Pause execution for debugging
    input("Press any key to continue...")
 
# Main function
def main():
    choice = int(input("Enter 1 to insert data from openpowerlifting.csv or 2 to insert data from openpowerlifting-2024-01-06-4c732975.csv: "))
 
    if choice == 1:
        dbname = "openpowerlifting"
        conn = connect_to_db(dbname)
        if conn:
            create_tables(conn, dbname)
            insert_data_openpowerlifting(conn)
            conn.close()
            print("Connection to database closed")
        else:
            print("Connection to database failed. Exiting...")
    elif choice == 2:
        dbname = "openpowerlifting-2024-01-06-4c732975"
        conn = connect_to_db(dbname)
        if conn:
            create_tables(conn, dbname)
            insert_data_openpowerlifting_2024(conn)
            conn.close()
            print("Connection to database closed")
        else:
            print("Connection to database failed. Exiting...")
    else:
        print("Invalid choice. Please enter either 1 or 2.")

if __name__ == "__main__":
    main()




#This take about an hour: This is the end result.