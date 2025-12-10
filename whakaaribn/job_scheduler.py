import subprocess
import time

import schedule


def job():
    subprocess.run(["/env/bin/whakaari_forecasts", "--sensitivity", "--ensemble", "--outdir", "/opt/data", "--log-level", "DEBUG"])


def main():
    job()
    #schedule.every().day.at("13:00").do(job)
    schedule.every().hour.do(job)

    while True:
        schedule.run_pending()
        time.sleep(60)
