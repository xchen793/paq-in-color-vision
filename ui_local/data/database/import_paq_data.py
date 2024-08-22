import json
import mysql.connector

try:
    connection = mysql.connector.connect(
        host='xchen920.mysql.pythonanywhere-services.com',
        user='xchen920',
        password='Cxz19980619*',
        database='xchen920$paqcolorvision'
    )
    if connection.is_connected():
        print("Connected to the database")
except mysql.connector.Error as err:
    print(f"Error: {err}")
    if err.errno == mysql.connector.errorcode.ER_ACCESS_DENIED_ERROR:
        print("Something is wrong with your username or password")
    elif err.errno == mysql.connector.errorcode.ER_BAD_DB_ERROR:
        print("Database does not exist")
    else:
        print(err)


def insert_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    print("JSON data loaded successfully")

    
    cursor = connection.cursor(dictionary=True)

    for user_id, user_data in data.items():
        # Insert user (if not already exists)
        cursor.execute(
            "INSERT INTO users (prolific_id) VALUES (%s) ON DUPLICATE KEY UPDATE prolific_id=prolific_id", 
            (user_id,)
        )
        
        # Insert survey pages for the user
        for page_name, page_data in user_data.items():
            cursor.execute("""
                INSERT INTO survey_pages 
                (prolific_id, survey_page, fixedColor_x, fixedColor_y, fixedColor_YY, query_vec_x, query_vec_y, query_vec_YY, gamma, endColor_x, endColor_y, endColor_YY, endColor_flag, endColor_query_vec_x, endColor_query_vec_y, endColor_query_vec_YY, time_taken) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE 
                    fixedColor_x=VALUES(fixedColor_x), fixedColor_y=VALUES(fixedColor_y), fixedColor_YY=VALUES(fixedColor_YY),
                    query_vec_x=VALUES(query_vec_x), query_vec_y=VALUES(query_vec_y), query_vec_YY=VALUES(query_vec_YY),
                    gamma=VALUES(gamma), endColor_x=VALUES(endColor_x), endColor_y=VALUES(endColor_y), endColor_YY=VALUES(endColor_YY),
                    endColor_flag=VALUES(endColor_flag), endColor_query_vec_x=VALUES(endColor_query_vec_x), endColor_query_vec_y=VALUES(endColor_query_vec_y), endColor_query_vec_YY=VALUES(endColor_query_vec_YY), time_taken=VALUES(time_taken)
                """, (
                    user_id,
                    page_name,
                    page_data['fixedColor']['x'], page_data['fixedColor']['y'], page_data['fixedColor']['Y'],
                    page_data['query_vec']['x'], page_data['query_vec']['y'], page_data['query_vec']['Y'],
                    page_data['gamma'],
                    page_data['endColor']['x'], page_data['endColor']['y'], page_data['endColor']['Y'],
                    page_data['endColor']['flag'],
                    page_data['endColor']['query_vec']['x'], page_data['endColor']['query_vec']['y'], page_data['endColor']['query_vec']['Y'],
                    page_data['timeTaken']
                )
            )

    connection.commit()
    cursor.close()
    connection.close()


if __name__ == "__main__":
    insert_data('color_data.json')
