from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from chatbot_api import ask_question

app = Flask(__name__)

@app.route("/whatsapp", methods=['POST'])
def whatsapp():
    incoming_msg = request.form.get('Body')
    response_text = ask_question(incoming_msg)
    resp = MessagingResponse()
    resp.message(response_text)
    return str(resp)

if __name__ == "__main__":
    app.run(debug=True)
