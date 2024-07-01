from flask import Flask, request, jsonify
import json
from main import save_embedding_to_chromadb, collection

with open('data.json', 'r') as file:
    data = json.load(file)

app = Flask(__name__)

@app.route('/get_values', methods=['POST'])
def get_values():
    if request.is_json:
        #data = request.get_json()
        
        assigned_task = data.get('assignedTask', {})
        assigned_task_desc = assigned_task.get('taskDesc')
        
        users = data.get('users', [])

        for user in users:
            empid = user.get('empid')
            tasks = user.get('tasks', [])
            for task in tasks:
                task_id=task.get('taskId')
                taskdescription = task.get('taskdescription')

        for user in users:
            for task in tasks:
                save_embedding_to_chromadb(task)
        
        results = collection.query(query_texts=[assigned_task_desc], n_results=3)

        # Return an empty JSON response
        return jsonify({}), 200
    else:
        return jsonify({"error": "Request must be JSON"}), 400

if __name__ == '__main__':
    app.run(debug=True)