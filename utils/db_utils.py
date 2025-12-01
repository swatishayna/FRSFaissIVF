import pymysql
import os
import shutil
import json
import base64

from utils.config_utils import load_config
import logging

logger = logging.getLogger(__name__)

def get_connection():
    """
    Establish MySQL connection using configuration from YAML.
    """
    config = load_config()
    db_config = config['db']

    print(f"🔍 Trying to connect to DB at {db_config['host']}:{db_config['port']} ...")

    try:
        conn = pymysql.connect(
            host=db_config['host'],
            user=db_config['user'],
            password=db_config['password'],
            database=db_config['database'],
            port=db_config['port'],
            cursorclass=pymysql.cursors.DictCursor
        )
        print("✅ Database connection successful.")
        return conn
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        raise e

def fetch_images_from_db(conn, batch_size=5000, last_id=0):
    query = f"""
    SELECT
        taf.ACCUSED_FILE_SRNO AS FILE_SRNO,
        tai.ACCUSED_SRNO,
        taf.FILE_NAME,
        CONCAT(tai.FIRST_NAME, ' ', tai.LAST_NAME) AS ACCUSED_NAME,
        tai.RELATIVE_NAME,
        tai.AGE,
        mg.GENDER,
        CONCAT(SUBSTR(tfr.FIR_REG_NUM, -4), '/', tfr.REG_YEAR) AS FIR_REG_NUM,
        tai.AGE,
        tai.YOB,
        d.DISTRICT,
        p.PS,
        taf.UPLOADED_FILE
    FROM t_accused_info tai
    LEFT JOIN t_fir_registration tfr ON tai.fir_reg_num = tfr.fir_reg_num AND tfr.LANG_CD=6 AND tfr.RECORD_STATUS <>'D'  
    LEFT JOIN t_accused_files taf ON taf.ACCUSED_SRNO = tai.ACCUSED_SRNO
    LEFT JOIN m_gender mg ON tai.GENDER_CD = mg.GENDER_CD AND mg.LANG_CD=6
    LEFT JOIN m_district d ON tfr.DISTRICT_CD = d.DISTRICT_CD AND d.LANG_CD = 6
    LEFT JOIN m_police_station p ON tfr.PS_CD = p.PS_CD AND p.LANG_CD = 6 
    WHERE taf.UPLOADED_FILE IS NOT NULL
    AND taf.ACCUSED_FILE_SRNO > {last_id}
    ORDER BY taf.ACCUSED_FILE_SRNO
    LIMIT {batch_size};
"""

    with conn.cursor(pymysql.cursors.DictCursor) as cursor:
        logger.info(f"⏳ Fetching batch from DB > {last_id}")
        cursor.execute(query)
        logger.info(f"✅ Fetched rows")
        return cursor.fetchall()


# def fetch_images_from_db(conn, query=None, batch_size=5000, offset=0):
#     """
#     Fetch a batch of images from MySQL using the query defined in config.yaml.
#     Uses LIMIT/OFFSET for incremental batch loading.
#     """
#     config = load_config()
#     if query is None:
#         query = config['db']['query']

#     paginated_query = f"{query} LIMIT {batch_size} OFFSET {offset}"

#     try:
#         with conn.cursor() as cursor:
#             cursor.execute(paginated_query)
#             rows = cursor.fetchall()
#             print(f"📦 Retrieved {len(rows)} rows (offset={offset})")
#             return rows
#     except Exception as e:
#         print(f"⚠️ Database fetch failed: {e}")
#         return []

def get_total_row_count():
    """
    Returns the total number of rows in the source table defined in config.yaml.
    """
    config = load_config()
    db_config = config['db']

    conn = None
    try:
        conn = pymysql.connect(
            host=db_config['host'],
            user=db_config['user'],
            password=db_config['password'],
            database=db_config['database'],
            port=db_config['port'],
            cursorclass=pymysql.cursors.DictCursor
        )

        # Derive table name from query if available
        query = db_config.get("query", "").lower()
        table_name = None
        if "from" in query:
            try:
                table_name = query.split("from")[1].split()[0]
            except Exception:
                table_name = None

        with conn.cursor() as cursor:
            if table_name:
                cursor.execute(f"SELECT COUNT(*) AS total FROM {table_name}")
            else:
                cursor.execute("SELECT COUNT(*) AS total FROM (" + query + ") AS subquery")
            result = cursor.fetchone()
            return result["total"]

    except Exception as e:
        print(f"⚠️ Failed to get total row count: {e}")
        return None
    finally:
        if conn:
            conn.close()

def save_n_images_from_db(n=10, output_dir="image/mysqldata/images"):
    """
    Fetches 'n' images using existing db_utils functions and saves them as .jpg files.
    """
    # Ensure output folder exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"📂 Saving images to: {os.path.abspath(output_dir)}")

    # Get database connection using your existing function
    conn = get_connection()

    # Reuse the same fetch function — we can just set batch_size = n and offset = 0
    rows = fetch_images_from_db(conn, batch_size=n, offset=0)
    if not rows:
        print("⚠️ No rows fetched from database.")
        conn.close()
        return

    print(f"📦 Retrieved {len(rows)} rows from database. Saving to disk...")

    for i, row in enumerate(rows, start=1):
        file_srno = row.get("FILE_SRNO", f"img_{i}")
        img_blob = row.get("UPLOADED_FILE")

        if not img_blob:
            print(f"⚠️ Skipping FILE_SRNO={file_srno} — No image blob found.")
            continue

        file_path = os.path.join(output_dir, f"{file_srno}.jpg")

        # Save image as binary file
        with open(file_path, "wb") as f:
            f.write(img_blob)

        print(f"✅ Saved: {file_path}")

    # Close connection
    conn.close()
    print("🔌 Database connection closed.")

def fetch_n_rows_with_columns(n=10):
    """
    Fetch n rows from the configured table using existing DB utilities.
    Returns:
        columns (list): List of column names.
        rows (list of dicts): Data rows.
    """
    config = load_config()
    db_query = config["db"].get("query", "")

    conn = None
    try:
        conn = get_connection()
        paginated_query = f"{db_query} LIMIT {n}"

        with conn.cursor() as cursor:
            cursor.execute(paginated_query)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description] if cursor.description else []

        print(f"📊 Retrieved {len(rows)} rows with {len(columns)} columns.")
        return columns, rows

    except Exception as e:
        print(f"⚠️ Failed to fetch sample rows: {e}")
        return [], []
    finally:
        if conn:
            conn.close()
            print("🔌 Database connection closed.")

def fetch_n_rows_with_columns(n=10):
    """
    Fetch n rows from the configured table using existing DB utilities.
    Returns:
        columns (list): List of column names.
        rows (list of dicts): Data rows.
    """
    config = load_config()
    db_query = config["db"].get("query", "")

    conn = None
    try:
        conn = get_connection()
        paginated_query = f"{db_query} LIMIT {n}"

        with conn.cursor() as cursor:
            cursor.execute(paginated_query)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description] if cursor.description else []

        print(f"📊 Retrieved {len(rows)} rows with {len(columns)} columns.")
        return columns, rows

    except Exception as e:
        print(f"⚠️ Failed to fetch sample rows: {e}")
        return [], []
    finally:
        if conn:
            conn.close()
            print("🔌 Database connection closed.")

# total_rows = get_total_row_count()
# print(total_rows)

def fetch_face_records(batch_size=5000):
    """
    Fetch all face records (with image BLOBs and metadata) from MySQL in batches.
    Uses query defined in config.yaml.
    
    Returns:
        all_rows (list[dict]): Combined list of records across all batches.
    """
    conn = get_connection()
    all_rows = []
    offset = 0

    try:
        while True:
            rows = fetch_images_from_db(conn, batch_size=batch_size, offset=offset)
            if not rows:
                break
            all_rows.extend(rows)
            offset += batch_size

            print(f"✅ Fetched {len(rows)} rows, total accumulated: {len(all_rows)}")

        print(f"📦 Finished fetching total {len(all_rows)} records from DB.")
        return all_rows

    except Exception as e:
        print(f"❌ Error in fetch_face_records: {e}")
        return []
    finally:
        conn.close()
        print("🔌 Database connection closed.")

def get_image_row_by_filesrno(file_srno, image_base_path):
    """
    Fetch DB row using FILE_SRNO
    Save image BLOB locally
    Convert saved image to Base64
    Return clean response dictionary (NO BYTES)
    """

    conn = get_connection()
    # -------------------------------------------------------------
    # 1️⃣ Clear the image folder
    # -------------------------------------------------------------
    if os.path.exists(image_base_path):
        shutil.rmtree(image_base_path)
    os.makedirs(image_base_path, exist_ok=True)

    # -------------------------------------------------------------
    # 2️⃣ Fetch row
    # -------------------------------------------------------------
    query = """
        SELECT
            taf.ACCUSED_FILE_SRNO AS FILE_SRNO,
            tai.ACCUSED_SRNO,
            taf.FILE_NAME,
            CONCAT(tai.FIRST_NAME, ' ', tai.LAST_NAME) AS ACCUSED_NAME,
            tai.RELATIVE_NAME,
            tai.AGE,
            mg.GENDER,
            CONCAT(SUBSTR(tfr.FIR_REG_NUM, -4), '/', tfr.REG_YEAR) AS FIR_REG_NUM,
            taf.UPLOADED_FILE
        FROM t_accused_info tai
        JOIN t_fir_registration tfr ON tai.fir_reg_num = tfr.fir_reg_num
        JOIN t_accused_files taf ON taf.ACCUSED_SRNO = tai.ACCUSED_SRNO
        JOIN m_gender mg ON tai.GENDER_CD = mg.GENDER_CD AND mg.LANG_CD = 6
        WHERE mg.GENDER_CD = 3
          AND tfr.REG_YEAR IN (2024, 2025)
          AND taf.ACCUSED_FILE_SRNO = %s;
    """

    with conn.cursor(pymysql.cursors.DictCursor) as cursor:
        cursor.execute(query, (file_srno,))
        rows = cursor.fetchall()
        if not rows:
            return None

    row = rows[0]

    # -------------------------------------------------------------
    # 3️⃣ Extract and save image
    # -------------------------------------------------------------
    image_blob = row.get("UPLOADED_FILE")

    base64_image = None
    image_path = None

    if image_blob:
        # Save file locally
        file_path = os.path.join(image_base_path, row["FILE_NAME"])
        try:
            with open(file_path, "wb") as f:
                f.write(image_blob)
            image_path = file_path
        except:
            pass

        # Convert to Base64
        try:
            base64_image = base64.b64encode(image_blob).decode("utf-8")
        except:
            base64_image = None

    # -------------------------------------------------------------
    # 4️⃣ Return clean, JSON-safe dict
    # -------------------------------------------------------------
    response = {
        "FILE_SRNO": row.get("FILE_SRNO"),
        "ACCUSED_SRNO": row.get("ACCUSED_SRNO"),
        "FILE_NAME": row.get("FILE_NAME"),
        "ACCUSED_NAME": row.get("ACCUSED_NAME"),
        "RELATIVE_NAME": row.get("RELATIVE_NAME"),
        "AGE": row.get("AGE"),
        "GENDER": row.get("GENDER"),
        "FIR_REG_NUM": row.get("FIR_REG_NUM"),
        "image_path": image_path,
        "image_base64": base64_image
    }

    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(response, indent=4, ensure_ascii=False))
    return response

import os
import shutil
import json
import base64
import pymysql
from utils.db_utils import get_connection


def get_image_row_by_filesrno_new(file_srno, image_base_path):
    """
    Fetch DB row using FILE_SRNO (no filters)
    Save image BLOB locally
    Convert saved image to Base64
    Return clean response dictionary
    """

    conn = get_connection()

    # -------------------------------------------------------------
    # 1️⃣ Clear image folder
    # -------------------------------------------------------------
    if os.path.exists(image_base_path):
        shutil.rmtree(image_base_path)
    os.makedirs(image_base_path, exist_ok=True)

    # -------------------------------------------------------------
    # 2️⃣ Query WITHOUT FILTERS
    # -------------------------------------------------------------
    query = """
        SELECT
            taf.ACCUSED_FILE_SRNO AS FILE_SRNO,
            tai.ACCUSED_SRNO,
            taf.FILE_NAME,
            CONCAT(tai.FIRST_NAME, ' ', tai.LAST_NAME) AS ACCUSED_NAME,
            tai.RELATIVE_NAME,
            tai.AGE,
            mg.GENDER,
            CONCAT(SUBSTR(tfr.FIR_REG_NUM, -4), '/', tfr.REG_YEAR) AS FIR_REG_NUM,
            taf.UPLOADED_FILE
        FROM t_accused_info tai
        JOIN t_fir_registration tfr ON tai.fir_reg_num = tfr.fir_reg_num
        JOIN t_accused_files taf ON taf.ACCUSED_SRNO = tai.ACCUSED_SRNO
        LEFT JOIN m_gender mg ON tai.GENDER_CD = mg.GENDER_CD AND mg.LANG_CD = 6
        WHERE taf.ACCUSED_FILE_SRNO = %s;
    """

    with conn.cursor(pymysql.cursors.DictCursor) as cursor:
        cursor.execute(query, (file_srno,))
        rows = cursor.fetchall()
        if not rows:
            return {"error": f"No record found for FILE_SRNO={file_srno}"}

    row = rows[0]

    # -------------------------------------------------------------
    # 3️⃣ Save Image (if exists)
    # -------------------------------------------------------------
    image_blob = row.get("UPLOADED_FILE")
    image_path = None
    base64_image = None

    if image_blob:
        file_path = os.path.join(image_base_path, row["FILE_NAME"])

        # Save file
        try:
            with open(file_path, "wb") as f:
                f.write(image_blob)
            image_path = file_path
        except Exception:
            pass

        # Base64 encode
        try:
            base64_image = base64.b64encode(image_blob).decode("utf-8")
        except Exception:
            base64_image = None

    # -------------------------------------------------------------
    # 4️⃣ Build clean response
    # -------------------------------------------------------------
    response = {
        "FILE_SRNO": row.get("FILE_SRNO"),
        "ACCUSED_SRNO": row.get("ACCUSED_SRNO"),
        "FILE_NAME": row.get("FILE_NAME"),
        "ACCUSED_NAME": row.get("ACCUSED_NAME"),
        "RELATIVE_NAME": row.get("RELATIVE_NAME"),
        "AGE": row.get("AGE"),
        "GENDER": row.get("GENDER"),
        "FIR_REG_NUM": row.get("FIR_REG_NUM"),
        "image_path": image_path,
        "image_base64": base64_image
    }

    # Debug output
    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(response, indent=4, ensure_ascii=False))

    return response


def get_image_row_by_filesrno_check(file_srno, image_base_path):
    """
    Fetch DB row using FILE_SRNO (no filters)
    Save image BLOB locally
    Convert saved image to Base64
    Return clean response dictionary
    """

    conn = get_connection()

    # -------------------------------------------------------------
    # 1️⃣ Clear image folder
    # -------------------------------------------------------------
    if os.path.exists(image_base_path):
        shutil.rmtree(image_base_path)
    os.makedirs(image_base_path, exist_ok=True)

    # -------------------------------------------------------------
    # 2️⃣ Query WITHOUT FILTERS
    # -------------------------------------------------------------
    query = """
      
        SELECT
            taf.ACCUSED_FILE_SRNO AS FILE_SRNO,
            tai.ACCUSED_SRNO,
            taf.FILE_NAME,
            CONCAT(tai.FIRST_NAME, ' ', tai.LAST_NAME) AS ACCUSED_NAME,
            tai.RELATIVE_NAME,
            tai.AGE,
            mg.GENDER,
            CONCAT(SUBSTR(tfr.FIR_REG_NUM, -4), '/', tfr.REG_YEAR) AS FIR_REG_NUM,
            d.DISTRICT,
            p.PS,
            taf.UPLOADED_FILE
        FROM t_accused_info tai
        JOIN t_fir_registration tfr 
            ON tai.fir_reg_num = tfr.fir_reg_num
        JOIN t_accused_files taf 
            ON taf.ACCUSED_SRNO = tai.ACCUSED_SRNO
        LEFT JOIN m_gender mg 
            ON tai.GENDER_CD = mg.GENDER_CD AND mg.LANG_CD = 6
        LEFT JOIN m_district d
            ON tfr.DISTRICT_CD = d.DISTRICT_CD AND d.LANG_CD = 6
        LEFT JOIN m_police_station p
            ON tfr.PS_CD = p.PS_CD AND p.LANG_CD = 6
        WHERE taf.ACCUSED_FILE_SRNO = %s;
    """


    with conn.cursor(pymysql.cursors.DictCursor) as cursor:
        cursor.execute(query, (file_srno,))
        rows = cursor.fetchall()
        if not rows:
            return {"error": f"No record found for FILE_SRNO={file_srno}"}

    row = rows[0]

    # -------------------------------------------------------------
    # 3️⃣ Save Image (if exists)
    # -------------------------------------------------------------
    image_blob = row.get("UPLOADED_FILE")
    image_path = None
    base64_image = None

    if image_blob:
        file_path = os.path.join(image_base_path, row["FILE_NAME"])

        # Save file
        try:
            with open(file_path, "wb") as f:
                f.write(image_blob)
            image_path = file_path
        except Exception:
            pass

        # Base64 encode
        try:
            base64_image = base64.b64encode(image_blob).decode("utf-8")
        except Exception:
            base64_image = None

    # -------------------------------------------------------------
    # 4️⃣ Build clean response
    # -------------------------------------------------------------
    response = {
        "FILE_SRNO": row.get("FILE_SRNO"),
        "ACCUSED_SRNO": row.get("ACCUSED_SRNO"),
        "FILE_NAME": row.get("FILE_NAME"),
        "ACCUSED_NAME": row.get("ACCUSED_NAME"),
        "RELATIVE_NAME": row.get("RELATIVE_NAME"),
        "AGE": row.get("AGE"),
        "GENDER": row.get("GENDER"),
        "FIR_REG_NUM": row.get("FIR_REG_NUM"),
        "DISTRICT": row.get("DISTRICT"),
        "POLICE_STATION": row.get("PS"),
        "image_path": image_path,
        "image_base64": base64_image
    }

    # Debug output
    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(response, indent=4, ensure_ascii=False))

    return response



# conn = get_connection()
file_srno = int("333030062500004592")
# file_srno = int("3330100123000172601")

image_base_path = "temp_images"
row_data = get_image_row_by_filesrno_check(file_srno=file_srno, image_base_path=image_base_path)
# print("This",row_data.get("POLICE_STATION"))
# # # with open("output.txt", "w", encoding="utf-8") as f:
# # #     f.write(json.dumps(row_data, indent=4, ensure_ascii=False))

# print(int("3336504423000810801"))






def serialize_row(row):
    """Convert bytes → base64 strings inside a row dict."""
    serialized = {}
    for key, value in row.items():
        if isinstance(value, bytes):
            # Convert bytes to base64 string
            serialized[key] = base64.b64encode(value).decode("utf-8")
        else:
            serialized[key] = value
    return serialized


# r, c = fetch_n_rows_with_columns(n=1)

# # Convert all rows
# clean_rows = [serialize_row(row) for row in c]

