from bs4 import BeautifulSoup
import requests
import json
import os
import time

scraperapi_api_key = os.getenv("SCRAPERAPI_API_KEY")

class Scraper():
    def scrape(self, url: str) -> str:
        print(f"scraping {url} content".format(url))

        """Issue request via scraperAPI"""
        def issueRequest():
            headers = {
                "User-Agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Mobile Safari/537.36",
                "accept-encoding": "gzip, deflate, br",
                "accept-language": "pl-PL,pl;q=0.9,en-US;q=0.8,en;q=0.7",
            }
        
            try:
                async_job = requests.post(
                    url = 'https://async.scraperapi.com/jobs',
                    json = {
                        'apiKey': scraperapi_api_key,
                        'url': url
                    }
                )

                return async_job.json()   
            except Error:
                print(f"Error occurred when parsing data: {e.msg}")
                pass    

        work_in_progress = issueRequest()
        if not work_in_progress:
            raise ValueError("Failed to initiate scraping job")

        scraped = None
        while work_in_progress['status'] == 'running':
            print("waiting for scraping job to complete..")
            time.sleep(1) #sleep 1sec
            req = requests.get(work_in_progress['statusUrl'])
            res = req.json()
            job_status = res['status']
            if job_status == 'finished':
                scraped = res['response']['body']
                break

        soup = BeautifulSoup(scraped, "html.parser")

        return soup

