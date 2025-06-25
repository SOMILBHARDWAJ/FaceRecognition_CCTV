from datetime import date

from DatabaseModel.DBInstanceProvider import get_connection


def add_user_with_id(user_id, name, encoding):
    conn = get_connection()
    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO users (user_id, name, face_encoding)
            VALUES (%s, %s, %s);
            """,
            (user_id, name, encoding)
        )
        conn.commit()
        return True
def get_user_encodings():
    conn = get_connection()
    with conn.cursor() as cursor:
        cursor.execute("SELECT user_id, face_encoding FROM users;")
        rows = cursor.fetchall()

    # Convert to dictionary
    encoding_map = {user_id: encoding for user_id, encoding in rows}
    return encoding_map
def get_user_name_by_id(user_id):
    conn = get_connection()
    with conn.cursor() as cursor:
        cursor.execute("SELECT name FROM users WHERE user_id = %s;", (user_id,))
        result = cursor.fetchone()
        return result[0] if result else None

def upsert_attendance(user_id, status, time_in=None, time_out=None, worked_time=None, absent_time=None):
    conn = get_connection()
    today = date.today()
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO attendance (user_id, date, status, time_in, time_out, worked_time, absent_time)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (user_id, date)
            DO UPDATE SET
                status = EXCLUDED.status,
                time_in = EXCLUDED.time_in,
                time_out = EXCLUDED.time_out,
                worked_time = EXCLUDED.worked_time,
                absent_time = EXCLUDED.absent_time;
        """, (user_id, today, status, time_in, time_out, worked_time, absent_time))
        conn.commit()

