import subprocess
import smtplib
from email.message import EmailMessage
from datetime import datetime
import sys
from pipeline.config import EMAIL_PASS, EMAIL_RECEIVER, EMAIL_SENDER

def send_email(subject, body):
    msg = EmailMessage()
    msg.set_content(body)
    msg['Subject'] = subject
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASS)
            server.send_message(msg)
        print(f"Email sent: {subject}")
    except Exception as e:
        print(f"Failed to send email: {str(e)}")

def run_step(step_name, command):
    print(f"\n--- Running Step: {step_name} ---")
    start_time = datetime.now()
    
    output_log = []
    
    process = subprocess.Popen(
        command, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True,
        bufsize=1
    )
    
    for line in process.stdout:
        print(line, end="")
        output_log.append(line)
        
    process.wait()
    
    end_time = datetime.now()
    duration = end_time - start_time
    full_output = "".join(output_log)
    
    if process.returncode != 0:
        error_log = full_output[-2000:] if full_output else "No output."
        error_msg = f"Step '{step_name}' failed at {end_time}.\n\nLog Tail:\n{error_log}"
        
        send_email(f"Pipeline Failed: {step_name}", error_msg)
        print(f"\nPipeline Failed due to error in {step_name}.")
        sys.exit(1) 
    else:
        success_log = full_output[-1000:] if full_output else "No output."
        success_msg = f"Step '{step_name}' completed successfully in {duration}.\n\nLog Tail:\n{success_log}"
        
        send_email(f"Pipeline succeeded Step: {step_name}", success_msg)

if __name__ == "__main__":
    print("Starting Training Pipeline Automation")
    
    run_step("1_Encoding", "python3 encode_all.py")
    run_step("2_Combining_Data", "python3 combine.py")
    run_step("3_Model_Training", "python3 train.py")
