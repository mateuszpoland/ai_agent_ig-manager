# AI Agent Instagram Manager

This project is designed to automate the management of an Instagram car fan page using an AI agent. The AI agent is capable of generating engaging posts, calculating car-related costs, and more.

## Setup

### Prerequisites

- Docker
- Docker Compose

### Environment Variables

Create a `.env` file in the main directory with the following content:

OPENAI_API_KEY=
SERPER_API_KEY=
SCRAPERAPI_API_KEY=
EXCHANGERATE_API_KEY=

and plug necessary API keys.

### Docker Setup

To set up the project using Docker, follow these steps:

1. **Build and Start the Containers**

   Run the following command to build and start the containers:

   ```
   docker-compose up --build
   ```
2. **Running the Agent**
   The manager.py script is the entry point to run the AI agent. It fetches the next post data from the MongoDB database, runs the agent to generate a post, and saves the generated content to the filesystem.

   ```
    docker-compose run app python ig_manager/manager.py
   ```

### How It Works

**Data Storage:** Car-related data is stored in a MongoDB database.
**Agent Execution:** The manager.py script fetches the data for the next scheduled post and invokes the AI agent to generate content.
**Content Generation:** The AI agent uses various tools to gather necessary information, calculate costs, and generate an engaging 
**Result Storage:** The generated post is saved to the filesystem and marked as approved in the database.

Tools and Libraries Used:
    **Langchain:** For creating and managing AI agents.
    **Pymongo:** For interacting with MongoDB.
    **BeautifulSoup:** For web scraping.
    **Docker:** For containerization.
    **MongoDB:** For data storage.
    **Mongo Express:** For MongoDB administration.
