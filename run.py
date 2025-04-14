from flask import render_template, request, jsonify, redirect, url_for
import requests
from flask_server import app, db
import flask_server.university
from flask_server.university.models import Holidays, Course, Student, Teacher
from chat import chatbot_response
from flask_server.university.nlp_utils import course_matcher
import traceback

with app.app_context():
    db.create_all()

@app.post("/chatbot_api/")
def normal_chat():
    try:
        msg = request.get_json().get('message')
        if not msg:
            return jsonify({'response': 'No message provided', 'tag': 'error'}), 400

        response, tag = chatbot_response(msg)

        if tag == 'result':
            return jsonify({'response': response, 'tag': tag, 'url': 'result/'})

        elif tag == 'courses':
            course = course_matcher(msg)
            if course:
                course_details = Course.query.filter_by(name=course).first()
                if course_details:
                    response = f"{course_details.name} takes {course_details.duration}"
                    link = f"{request.host_url}courses/syllabus/{course_details.id}/"
                    return jsonify({
                        'response': response,
                        'tag': tag,
                        'data': {
                            'filename': f"{course_details.name} syllabus",
                            'link': link
                        }
                    })
                else:
                    response = "Course not found."
            else:
                all_courses = Course.query.all()
                response = "Here are the available courses:\n"
                for course in all_courses:
                    response += f"\n{course.name}"

        elif tag == "holidays":
            holiday = Holidays.query.first()
            if holiday:
                link = f"{request.host_url}holidays/download/{holiday.id}/"
                response = f"Holidays for year {holiday.year} are available for download."
                return jsonify({
                    'response': response,
                    'tag': tag,
                    'data': {
                        'filename': holiday.file_name,
                        'link': link
                    }
                })
            else:
                response = "No holiday data found."

      elif tag == 'faculty': # Start
    try:
        url = f"{request.host_url}teachers/api/"
        data = requests.get(url=url)  # ✅ Indented correctly

        if data.status_code == 200:
            response += "\n" + "\n".join(
                f"{item['name']} ({item['department']})" for item in data.json()
            )
        else:
            response += "\nError fetching faculty data."
    except Exception as e:
        response += f"\nFailed to fetch faculty info: {str(e)}" #End


        return jsonify({'response': response, 'tag': tag})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'response': f"Server error: {str(e)}", 'tag': 'error'}), 500


@app.post("/chatbot_api/result/")
def fetch_result():
    try:
        msg = request.get_json().get('message')
        if not msg:
            return jsonify({'response': 'No student ID provided', 'url': ''}), 400

        studentID = msg.strip()

        if not studentID.isdigit():
            return jsonify({'response': "Please use the correct format: e.g., 434121010021", 'url': 'result/'})

        student = Student.query.get(studentID)

        if student:
            response = f"Result of {studentID} is {student.cgpa}"
            url = ""
        else:
            return jsonify({'response': "Student not found", 'url': ""})

        return jsonify({'response': response, 'url': url})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'response': f"Error: {str(e)}", 'url': ''}), 500

