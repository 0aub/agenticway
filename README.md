# AgenticWay

This repository is under development, providing various agentic applications for your daily work.

## Running the Slide to Video Application

### Step 1: Configure

Edit the `./domains/study/slide_to_vid/config.yaml` file to specify the files to process and the experiment name:

```yaml
exp_name: 'your_experiment_name'
files:
  - data/Lecture-1.pdf
  - data/Lecture-2.pdf

```

### Step 2: Build the Docker Image

Run the following command to build the Docker image:

```bash
docker compose -f docker-compose.study.yml build
```

### Step 3: Start the Application

Use this command to run the application:

```bash
docker compose -f docker-compose.study.yml up
```

This will process the specified PDF files and generate narrated videos.

