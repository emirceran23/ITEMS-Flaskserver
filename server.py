from flask_socketio import SocketIO, emit, join_room, leave_room, disconnect
import time, random, threading
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
import base64
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename
from mainFlask import analyze_image  # your analysis function
from pdf_report_generator import PDFReportGenerator  # your report generator module
from flask import jsonify

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Global variables
activeClients = {}       # { sid: { 'deviceType': ..., 'lastHeartbeat': ..., 'roomCode': ... } }
rooms = {}               # { roomCode: { 'clients': [sid, ...], 'testType': 'original', 'testRoom': testRoom } }
roomTimers = {}          # { roomCode: timer_object }
finalLineScores = {}     # { roomCode: { userId: { lineIndex: score, ... } } }
lineLengths = [1, 2, 3, 4, 5, 6, 7, 8, 8, 8]
lineLengthsModified = [2, 3, 4, 5, 6, 8, 10, 10]
userAnswers = {}         # { roomCode: { userId: [ [None,...], ... ] } }
userProgress = {}        # { roomCode: { userId: { 'lineIndex': int, 'letterIndex': int } } }
sharedVariable = {}      # { roomCode: lineIndex }
mobileId = None          # Global mobile user id (if needed)
testFinished = False     # Global flag for finishing test

@app.route("/")
def index():
    return "Socket.IO server is running."

# Connection and Heartbeat
@socketio.on('connect')
def handle_connect():
    sid = request.sid
    print("A user connected: " + sid)
    activeClients[sid] = {'deviceType': 'unknown', 'lastHeartbeat': time.time(), 'roomCode': None}

@socketio.on('heartbeat')
def handle_heartbeat():
    sid = request.sid
    if sid in activeClients:
        activeClients[sid]['lastHeartbeat'] = time.time()
        print(f"Heartbeat received from {sid}")

# Authentication
@socketio.on('authenticate')
def handle_authenticate(data):
    sid = request.sid
    username = data.get('username')
    password = data.get('password')
    print(f"Authentication attempt by {sid} with username: {username}")
    if username == 'admin' and password == '12345':
        emit('auth_response', {'success': True, 'message': 'Authentication successful'})
        print(f"Authentication successful for {sid}")
    else:
        emit('auth_response', {'success': False, 'message': 'Authentication failed'})
        print(f"Authentication failed for {sid}")
        disconnect()

# Restart and Device Type
@socketio.on('restartKiosk')
def handle_restartKiosk():
    print("restartKiosk event received from client")
    emit('restartKiosk', broadcast=True, include_self=False)

@socketio.on('deviceType')
def handle_deviceType(data):
    sid = request.sid
    activeClients[sid]['deviceType'] = data  # Expecting the device type as payload
    print(f"Device Type received from {sid}: {data}")

# Letter Selection and Click Events
@socketio.on('letterSelected')
def handle_letterSelected(data):
    sid = request.sid
    letterIndex = data
    print(f"Letter selected by {sid}: {letterIndex}")
    global testFinished
    if testFinished:
        letterIndex = 0
    emit('letterIndex', letterIndex, broadcast=True, include_self=False)

@socketio.on('clicked')
def handle_clicked(data):
    sid = request.sid
    print(f"Button clicked event received from {sid}: {data}")
    emit('click', data, broadcast=True, include_self=False)

# Test Type Change
@socketio.on('test')
def handle_test(test_type):
    sid = request.sid
    print(f"Test type received from {sid}: {test_type}")
    roomCode = activeClients[sid].get('roomCode')
    if roomCode:
        if roomCode not in rooms:
            rooms[roomCode] = {}
        rooms[roomCode]['testType'] = test_type

        # Reset user progress for all users in the room
        if roomCode in userProgress:
            for userId in list(userProgress[roomCode].keys()):
                resetUserProgress(roomCode, userId)

        print(f"User progress reset for room {roomCode} due to test type change")
        emit('testName', rooms[roomCode]['testType'], room=roomCode)
        emit('progressReset', {'message': 'User progress has been reset due to test type change'}, room=roomCode)
        print(f"Test type '{rooms[roomCode]['testType']}' broadcasted to room {roomCode}")
    else:
        emit('testName', test_type, broadcast=True)
        print(f"Test type '{test_type}' broadcasted globally")
    

# User Answer and Scoring
@socketio.on('userAnswer')
def handle_userAnswer(data):
    sid = request.sid
    answer = data.get('answer')
    roomCode = activeClients[sid].get('roomCode')
    userId = sid
    global mobileId
    mobileId = userId

    if not roomCode:
        print(f"Room code not assigned for socket {sid}")
        emit('error', {'message': 'Room does not exist or test has finished.'})
        return

    if roomCode not in rooms:
        print(f"Room {roomCode} does not exist!")
        emit('error', {'message': 'Room does not exist.'})
        return

    currentLineLengths = lineLengths if rooms[roomCode]['testType'] == "original" else lineLengthsModified

    # Initialize user data if not set
    if roomCode not in userProgress or userId not in userProgress[roomCode]:
        print(f"User progress not initialized for {userId} in room {roomCode}")
        initializeUserData(roomCode, userId)

    lineIndex = userProgress[roomCode][userId]['lineIndex']
    letterIndex = userProgress[roomCode][userId]['letterIndex']

    if answer not in [0, 1]:
        print(f"Invalid answer from {userId}: {answer}")
        emit('error', {'message': 'Invalid answer. Must be 0 or 1.'})
        return

    emit('userAnswer', {'answer': answer}, room=roomCode)
    emit('answerReceived', "yes", room=roomCode)
    print("Answer processed")

    # Update answer and shared variable
    userAnswers[roomCode][userId][lineIndex][letterIndex] = answer
    sharedVariable[roomCode] = lineIndex
    print(f"User {userId} in room {roomCode} answered {answer} at line {lineIndex}, letter {letterIndex}")

    # Move to next letter; if line is complete, calculate score
    letterIndex += 1
    if letterIndex >= currentLineLengths[lineIndex]:
        answersForLine = userAnswers[roomCode][userId][lineIndex]
        result = calculateScoreForLine(answersForLine, lineIndex, roomCode, userId)
        line_score = result['score']
        test_finished_line = result['testFinished']

        if roomCode not in finalLineScores:
            finalLineScores[roomCode] = {}
        if userId not in finalLineScores[roomCode]:
            finalLineScores[roomCode][userId] = {}
        finalLineScores[roomCode][userId][lineIndex] = line_score

        print(f"Final score for user {userId} on line {lineIndex} in room {roomCode}: {line_score}")

        emit('lineScore', {'lineIndex': lineIndex, 'score': line_score}, room=sid)
        emit('allUserScores', userAnswers[roomCode], room=roomCode)

        # End test if test finished or last line reached
        if test_finished_line or lineIndex >= len(currentLineLengths) - 1:
            print(f"Test finished for user {userId} in room {roomCode}")
            emit('testFinished', {
                'userId': userId,
                'totalScore': line_score,
                'detailedResults': userAnswers[roomCode][userId]
            }, room=roomCode)
            clearRoomData(roomCode)
            resetUserProgress(roomCode, userId)
            return
        else:
            lineIndex += 1
            letterIndex = 0
            print(f"User {userId} proceeds to line {lineIndex}")
            emit('nextLine', {'nextLineIndex': lineIndex}, room=sid)

    # Update progress
    userProgress[roomCode][userId]['lineIndex'] = lineIndex
    userProgress[roomCode][userId]['letterIndex'] = letterIndex

# Answer Editing and Navigation
@socketio.on('answerEdit')
def handle_answerEdit(data):
    sid = request.sid
    roomCode = activeClients[sid].get('roomCode')
    global mobileId
    mobileUserId = mobileId
    line = data.get('line')
    letter = data.get('letter')
    if roomCode and mobileUserId and roomCode in userProgress and mobileUserId in userProgress[roomCode]:
        userProgress[roomCode][mobileUserId]['lineIndex'] = line
        userProgress[roomCode][mobileUserId]['letterIndex'] = letter
        print(f"User {mobileUserId} progress updated to line {line}, letter {letter}")

@socketio.on('prevButton')
def handle_prevButton():
    sid = request.sid
    roomCode = activeClients[sid].get('roomCode')
    if roomCode:
        emit('prevButton', room=roomCode)

@socketio.on('nextButton')
def handle_nextButton():
    sid = request.sid
    roomCode = activeClients[sid].get('roomCode')
    if roomCode:
        emit('nextButton', room=roomCode)

@socketio.on('endTest')
def handle_endTest():
    sid = request.sid
    roomCode = activeClients[sid].get('roomCode')
    global mobileId
    mobileUserId = mobileId
    answersForLine = None
    lineIndex = 0
    totalScore = None

    if roomCode not in sharedVariable and userAnswers.get(roomCode, {}).get(mobileUserId) is not None:
        lineIndex = 0
        print(f"End test: userAnswers for {mobileUserId} in room {roomCode}: {userAnswers[roomCode][mobileUserId][lineIndex]}")
        answersForLine = userAnswers[roomCode][mobileUserId][lineIndex]
        totalScore = calculateScoreForLetterIndex(answersForLine, lineIndex, roomCode, mobileUserId)
    elif roomCode in sharedVariable and userAnswers.get(roomCode, {}).get(mobileUserId) is not None:
        lineIndex = sharedVariable[roomCode]
        print(f"End test: sharedVariable for room {roomCode} is {sharedVariable[roomCode]}")
        print(f"UserAnswers for {mobileUserId} at line {lineIndex}: {userAnswers[roomCode][mobileUserId][lineIndex]}")
        answersForLine = userAnswers[roomCode][mobileUserId][lineIndex]
        totalScore = calculateScoreForLetterIndex(answersForLine, lineIndex, roomCode, mobileUserId)
    else:
        totalScore = "20/0"

    emit('testFinished', {
        'mobileUserId': mobileUserId,
        'totalScore': totalScore,
        'detailedResults': userAnswers.get(roomCode, {}).get(mobileUserId, [])
    }, room=roomCode)
    print(f"Test finished for user {mobileUserId} in room {roomCode} with total score {totalScore}")
    clearRoomData(roomCode)
    resetUserProgress(roomCode, mobileUserId)
#strabismus
@socketio.on('detectStrabismus')
def handle_detectStrabismus(imageBytes):
    sid = request.sid
    print(f"Detect strabismus event received from {sid}")
    image_data = base64.b64decode(imageBytes)
    np_image = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    # Run your analysis function (make sure it works in web mode)
    results = analyze_image(image, display_in_tk=False, segmentation_method="YOLO")

    # Save debug image
    debug_filename = "debug_" + str(sid) + ".jpg"
    debug_filepath = os.path.join(app.config['UPLOAD_FOLDER'], debug_filename)
    cv2.imwrite(debug_filepath, results["debug_image"])

    # Generate PDF report
    pdf_filename = "report_" + str(sid) + ".pdf"
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)
    report_generator = PDFReportGenerator()
    report_generator.generate_report(results, output_path=pdf_path)

    # Return URLs
    debug_url = url_for('uploaded_file', filename=debug_filename, _external=True)
    pdf_url = url_for('uploaded_file', filename=pdf_filename, _external=True)
    emit('strabismusResult', {
        'debug_image_url': debug_url,
        'pdf_url': pdf_url
    }, room=sid)
# Miscellaneous events
@socketio.on('allZero')
def handle_allZero(data):
    sid = request.sid
    print(f"All zero event received from {sid}: {data}")
    emit('zero', data, broadcast=True, include_self=False)

@socketio.on('panel')
def handle_panel(data):
    sid = request.sid
    print(f"Panel name received from {sid}: {data}")
    emit('panelName', data, broadcast=True, include_self=False)

# Room management: Create, Delete, Join, and Leave
@socketio.on('createRoom')
def handle_createRoom(data):
    sid = request.sid
    testRoom = data  # expecting testRoom as payload
    print(f"{sid} is attempting to create a new room")
    createNewRoom(sid, testRoom)

@socketio.on('deleteRoom')
def handle_deleteRoom(data):
    sid = request.sid
    roomCode = data
    print(f"{sid} requested to delete room: {roomCode}")
    if roomCode in rooms:
        emit('roomDeleted', roomCode, room=roomCode)
        print(f"Room {roomCode} deleted by {sid}")
        emit('clicked', False, room=roomCode)
        if roomCode in roomTimers:
            roomTimers[roomCode].cancel()
            del roomTimers[roomCode]
        if roomCode in rooms:
            del rooms[roomCode]
        if roomCode in userAnswers:
            del userAnswers[roomCode]
        if roomCode in userProgress:
            del userProgress[roomCode]
    else:
        print(f"Room {roomCode} does not exist for deletion.")

@socketio.on('joinRoom')
def handle_joinRoom(data):
    sid = request.sid
    roomCode = data.get('roomCode')
    roomType = data.get('roomType')
    if roomCode in rooms:
        if len(rooms[roomCode]['clients']) >= 2:
            emit('joinError', 'Room is full')
            print(f"Room {roomCode} is full. {sid} cannot join.")
            return
        if rooms[roomCode]['testRoom'].strip() != roomType.strip():
            emit('joinError', 'Room type is different')
            print(f"Room {roomCode} type '{rooms[roomCode]['testRoom']}' does not match '{roomType}'")
            return
        else:
            rooms[roomCode]['clients'].append(sid)
            join_room(roomCode)
            # Fallback: if the client is not in activeClients, add it
            if sid not in activeClients:
                print(f"Client {sid} not found in activeClients. Adding client.")
                activeClients[sid] = {'deviceType': 'unknown', 'lastHeartbeat': time.time(), 'roomCode': None}
            activeClients[sid]['roomCode'] = roomCode
            initializeUserData(roomCode, sid)
            emit('testName', rooms[roomCode]['testType'], room=sid)
            emit('click', True, room=sid)
            emit('message', f"User {sid} joined room {roomCode}", room=roomCode)
            emit('joinSuccess', roomCode, room=sid)
            print(f"{sid} joined room {roomCode}")
    else:
        emit('joinError', 'Room does not exist')
        print(f"Room {roomCode} does not exist. {sid} cannot join.")


@socketio.on('leaveRoom')
def handle_leaveRoom():
    sid = request.sid
    roomCode = activeClients[sid].get('roomCode')
    userId = sid
    if not roomCode:
        print(f"No roomCode found for user {userId}")
        return
    print(f"{sid} is leaving room: {roomCode}")
    if roomCode in rooms:
        if userId in rooms[roomCode]['clients']:
            rooms[roomCode]['clients'].remove(userId)
        deviceType = activeClients[sid].get('deviceType', 'unknown')
        if len(rooms[roomCode]['clients']) > 0 and deviceType == 'desktop':
            print(f"Desktop client {userId} (creator) disconnected; closing room {roomCode}")
            closeRoom(roomCode)
        else:
            resetUserProgress(roomCode, userId)
            print(f"Mobile client {userId} disconnected from room {roomCode}, but room remains open.")

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    print("User disconnected: " + sid)
    if sid in activeClients:
        roomCode = activeClients[sid].get('roomCode')
        deviceType = activeClients[sid].get('deviceType', 'unknown')
        del activeClients[sid]
        if roomCode and roomCode in rooms:
            if sid in rooms[roomCode]['clients']:
                rooms[roomCode]['clients'].remove(sid)
            emit('clientDisconnected', {'socketId': sid}, room=roomCode)
            if len(rooms[roomCode]['clients']) > 0 and deviceType == 'desktop':
                print(f"Desktop client {sid} (creator) disconnected; closing room {roomCode}")
                closeRoom(roomCode)
            else:
                resetUserProgress(roomCode, sid)
                print(f"Mobile client {sid} disconnected from room {roomCode}, but room remains open.")

# Helper functions
def closeRoom(roomCode):
    if roomCode in rooms:
        for clientId in rooms[roomCode]['clients']:
            emit('roomClosed', {'message': 'Room closed by the desktop client disconnect'}, room=clientId)
            if clientId in activeClients:
                activeClients[clientId]['roomCode'] = None
        rooms[roomCode]['testType'] = 'original'
        clearRoomData(roomCode)
        print(f"Room {roomCode} and all client data have been removed.")

def initializeUserData(roomCode, userId):
    currentLineLengths = lineLengths if rooms[roomCode]['testType'] == "original" else lineLengthsModified
    if roomCode not in userAnswers:
        userAnswers[roomCode] = {}
    if userId not in userAnswers[roomCode]:
        userAnswers[roomCode][userId] = []
        for length in currentLineLengths:
            userAnswers[roomCode][userId].append([None] * length)
    if roomCode not in userProgress:
        userProgress[roomCode] = {}
    userProgress[roomCode][userId] = {'lineIndex': 0, 'letterIndex': 0}

def createNewRoom(sid, testRoom):
    roomCode = str(random.randint(100000, 999999))
    rooms[roomCode] = {'clients': [sid], 'testType': 'original', 'testRoom': testRoom}
    socketio.server.enter_room(sid, roomCode, namespace='/')
    activeClients[sid]['roomCode'] = roomCode
    socketio.emit('testName', rooms[roomCode]['testType'], room=sid)
    socketio.emit('roomCreated', roomCode, room=sid)
    print(f"Room {roomCode} created by {sid} with default test type 'original'")
    initializeUserData(roomCode, sid)
    timer = threading.Timer(30.0, room_expiration, args=(sid, testRoom, roomCode))
    timer.start()
    roomTimers[roomCode] = timer

def room_expiration(sid, testRoom, roomCode):
    with app.app_context():
        if roomCode in rooms and len(rooms[roomCode]['clients']) == 1:
            socketio.emit('roomExpired', {'message': f"Room {roomCode} expired due to inactivity."}, room=roomCode)
            clearRoomData(roomCode)
            createNewRoom(sid, testRoom)

def calculateScoreForLetterIndex(answersForLine, lineIndex, roomCode, userId):
    if not answersForLine or not isinstance(answersForLine, list):
        print("answersForLine is invalid:", answersForLine)
        return '20/200' if rooms[roomCode]['testType'] == "original" else '20/100'
    totalCorrect = sum(1 for answer in answersForLine if answer == 1)
    totalLetters = len(answersForLine)
    totalIncorrect = totalLetters - totalCorrect

    lineScores = {
        0: '20/0',
        1: '20/200',
        2: '20/100',
        3: '20/70',
        4: '20/50',
        5: '20/40',
        6: '20/30',
        7: '20/25',
        8: '20/20',
        9: '20/15',
        10: '20/10',
    }
    lineScoresModified = {
        0: '20/0',
        1: '20/100',
        2: '20/70',
        3: '20/50',
        4: '20/40',
        5: '20/30',
        6: '20/25',
        7: '20/20',
        8: '20/15',
    }
    currentLineScores = lineScores if rooms[roomCode]['testType'] == "original" else lineScoresModified

    if totalCorrect >= totalIncorrect and totalCorrect != totalLetters:
        score_str = currentLineScores.get(lineIndex + 1, "") + '-' + str(totalIncorrect)
    elif totalCorrect == totalLetters:
        score_str = currentLineScores.get(lineIndex + 1, "")
    elif totalIncorrect > totalCorrect and totalIncorrect != totalLetters:
        score_str = currentLineScores.get(lineIndex, "") + '+' + str(totalCorrect)
    elif totalIncorrect == totalLetters:
        if lineIndex == 0:
            score_str = currentLineScores.get(lineIndex, "")
        else:
            score_str = finalLineScores[roomCode][userId].get(lineIndex - 1, "")
    else:
        score_str = currentLineScores.get(lineIndex, "")
    return score_str

def calculateScoreForLine(answersForLine, lineIndex, roomCode, userId):
    totalCorrect = sum(1 for answer in answersForLine if answer == 1)
    totalIncorrect = sum(1 for answer in answersForLine if answer == 0)
    totalLetters = len(answersForLine)
    correctnessRate = (totalCorrect / totalLetters) * 100 if totalLetters > 0 else 0
    print(f"User answered {totalCorrect} out of {totalLetters} correctly on line {lineIndex} ({correctnessRate:.2f}% correct)")

    lineScores = {
        0: '20/0',
        1: '20/200',
        2: '20/100',
        3: '20/70',
        4: '20/50',
        5: '20/40',
        6: '20/30',
        7: '20/25',
        8: '20/20',
        9: '20/15',
        10: '20/10',
    }
    lineScoresModified = {
        0: '20/0',
        1: '20/100',
        2: '20/70',
        3: '20/50',
        4: '20/40',
        5: '20/30',
        6: '20/25',
        7: '20/20',
        8: '20/15',
    }
    currentLineScores = lineScores if rooms[roomCode]['testType'] == "original" else lineScoresModified

    if totalCorrect >= totalIncorrect and totalCorrect != totalLetters:
        score_str = currentLineScores.get(lineIndex + 1, "") + '-' + str(totalIncorrect)
    elif totalCorrect == totalLetters:
        score_str = currentLineScores.get(lineIndex + 1, "")
    elif totalIncorrect > totalCorrect and totalIncorrect != totalLetters:
        score_str = currentLineScores.get(lineIndex, "") + '+' + str(totalCorrect)
    elif totalIncorrect == totalLetters:
        if lineIndex == 0:
            score_str = currentLineScores.get(lineIndex, "")
        else:
            score_str = finalLineScores[roomCode][userId].get(lineIndex - 1, "")
    else:
        score_str = currentLineScores.get(lineIndex, "")

    test_finished = False
    if correctnessRate < 50:
        test_finished = True
        print(f"Test finished due to correctness rate less than 50% on line {lineIndex}")
        return {'score': score_str, 'testFinished': test_finished}
    print(f"Score calculated: {score_str}")
    return {'score': score_str, 'testFinished': test_finished}

def resetUserProgress(roomCode, userId):
    if roomCode in userAnswers and userId in userAnswers[roomCode]:
        del userAnswers[roomCode][userId]
    if roomCode in userProgress and userId in userProgress[roomCode]:
        del userProgress[roomCode][userId]
    print(f"User progress reset for user {userId} in room {roomCode}")

def clearRoomData(roomCode):
    if roomCode in rooms:
        socketio.emit('click', False, room=roomCode)
        for clientId in rooms[roomCode]['clients']:
            if clientId in activeClients:
                activeClients[clientId]['roomCode'] = None
        del rooms[roomCode]
    if roomCode in userAnswers:
        del userAnswers[roomCode]
    if roomCode in userProgress:
        del userProgress[roomCode]
    if roomCode in roomTimers:
        roomTimers[roomCode].cancel()
        del roomTimers[roomCode]

# Background task for heartbeat checks
def server_heartbeat():
    while True:
        now = time.time()
        timeout = 15  # seconds
        socketio.emit('serverHeartbeat', {'timestamp': now})    
        for sid in list(activeClients.keys()):
            if now - activeClients[sid]['lastHeartbeat'] > timeout:
                print(f"Client {sid} is unresponsive. Notifying room.")
                roomCode = activeClients[sid].get('roomCode')
                if roomCode:
                    socketio.emit('clientUnresponsive', {'socketId': sid, 'deviceType': activeClients[sid].get('deviceType')}, room=roomCode)
                del activeClients[sid]
        time.sleep(5)

socketio.start_background_task(server_heartbeat)
@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Run analysis
    results = analyze_image(filepath, display_in_tk=False, segmentation_method="YOLO")

    # Save debug image
    debug_filename = "debug_" + filename
    debug_filepath = os.path.join(app.config['UPLOAD_FOLDER'], debug_filename)
    cv2.imwrite(debug_filepath, results["debug_image"])

    # Generate PDF report
    pdf_filename = "report_" + os.path.splitext(filename)[0] + ".pdf"
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)
    report_generator = PDFReportGenerator()
    report_generator.generate_report(results, output_path=pdf_path)

    # Return URLs
    debug_url = url_for('uploaded_file', filename=debug_filename, _external=True)
    pdf_url = url_for('uploaded_file', filename=pdf_filename, _external=True)
    return jsonify({
        'debug_image_url': debug_url,
        'pdf_url': pdf_url
    })
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
