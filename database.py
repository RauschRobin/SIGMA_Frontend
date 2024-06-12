import sqlite3

class ImageDatabase:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute("CREATE TABLE IF NOT EXISTS images (id INTEGER PRIMARY KEY, drawing BLOB NOT NULL, generated_image BLOB NOT NULL, rating INTEGER)")

    def add_images_to_database(self, drawing, generated_image):
        self.cursor.execute("INSERT INTO images (drawing, generated_image) VALUES (?, ?)", (sqlite3.Binary(drawing), sqlite3.Binary(generated_image)))
        self.conn.commit()

    def add_rating(self, rating, drawing, generated_image):
        self.cursor.execute("UPDATE images SET rating=? WHERE drawing=? AND generated_image=?", (rating, sqlite3.Binary(drawing), sqlite3.Binary(generated_image)))
        self.conn.commit()
        return self.cursor.lastrowid
