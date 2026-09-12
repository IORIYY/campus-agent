import sqlite3
from langchain_core.tools import tool

DB_PATH = "schedule.db"


@tool
def query_schedule(day: str) -> str:
    """查询指定星期几的课程安排。参数 day 格式如：周一、周二、周三。"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT time, course, location, teacher FROM schedule WHERE day = ? ORDER BY time",
        (day,)
    )
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return f"{day}没有课程安排。"

    result = f"{day}的课程安排：\n"
    for time, course, location, teacher in rows:
        result += f"- {time} {course} @{location}（{teacher}）\n"
    return result


if __name__ == "__main__":
    print(query_schedule.invoke("周三"))