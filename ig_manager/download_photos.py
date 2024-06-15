import requests
import os
from dotenv import load_dotenv

load_dotenv()
serper_api_key = os.getenv("SERPER_API_KEY")

NUM_IMAGES = 2

def search_google_images(query, num_images=NUM_IMAGES):
    endpoint = "https://api.serpstack.com/search"
    params = {
        "access_key": serper_api_key,
        "query": query,
        "type": "images",
        "num": num_images
    }
    response = requests.get(endpoint, params=params)
    print(response.text)

    return response.json().get('image_results', [])

def download_image(url, folder_name, image_name):
    response = requests.get(url)
    if response.status_code == 200:
        with open(os.path.join(folder_name, image_name), 'wb') as file:
            file.write(response.content)



def downloadImages(car_name, num_images=NUM_IMAGES):
    query = f"{car_name} beautiful photos wallpaper uhd"
    print(f"Searching for images of {car_name}...")
    
    # Search for images
    images = search_google_images(query, num_images)
    
    # Create a directory for the car
    if not os.path.exists(car_name):
        os.makedirs(car_name)
    
    # Download images
    for idx, image in enumerate(images):
        image_url = image['image_src']
        download_image(image_url, car_name, f"{car_name}_{idx}.jpg")
        print(f"Downloaded image {idx+1} for {car_name}")


downloadImages("2023 Chevrolet Corvette C8")