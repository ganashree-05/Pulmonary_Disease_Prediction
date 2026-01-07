import sqlite3

# Replace with your actual database file
conn = sqlite3.connect('your_database.db')
cursor = conn.cursor()

# Add email column if it doesn't exist
try:
    cursor.execute("ALTER TABLE user ADD COLUMN email TEXT;")
    print("Email column added successfully!")
except sqlite3.OperationalError:
    print("Email column already exists.")

conn.commit()
conn.close()
