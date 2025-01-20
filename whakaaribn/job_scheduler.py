import subprocess
import time

import schedule


def job():
    subprocess.run(["/env/bin/whakaari_forecasts", "--outdir", "/opt/data"])


def main():
    job()
    schedule.every().day.at("13:00").do(job)

    while True:
        schedule.run_pending()
        time.sleep(60)
