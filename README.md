# 🤖⚙️ CodeCompanion

**CodeCompanion** is a Python tool designed to assist developers in optimizing their workflow. Leveraging the power of Large Language Models (LLMs), it provides features to break down coding tasks, analyze and explain existing code, and automatically generate documentation (docstrings) for functions, methods, and classes. The goal is to make programming projects more manageable, enhance code comprehension, and automate tedious yet essential tasks.

Whether you're tackling a complex feature, trying to understand unfamiliar code, or documenting your latest module, **CodeCompanion** offers intelligent assistance to boost your productivity and learning.

## Table of Contents

* [Motivation](#motivation)
* [Features](#features)
* [How It Works](#how-it-works)
* [Getting Started](#getting-started)

  * [Prerequisites](#prerequisites)
  * [Cloning the Repository](#cloning-the-repository)
  * [Configuration](#Configure-the-Environment-Variable)
  * [Installation](#How-to-Install-Dependencies)
* [Usage](#Demo-Video-of-the-Program-in-Action)

  * [Demo video](#Demo-Video-of-the-Program-in-Action)
* [License](https://github.com/Alejandro-Bravo2/proyectoDigitalizacion/blob/main/LICENSE)
* [Contributing](https://github.com/Alejandro-Bravo2/proyectoDigitalizacion/blob/main/CONTRIBUTING.md)

## Motivation

The motivation behind CodeCompanion stems from the common challenges developers face: breaking down large tasks, understanding complex code written by others (or by themselves after some time!), and the often-overlooked but crucial task of writing documentation.

This project was started with the goal of creating practical AI-powered tools that directly address these pain points. By automating task breakdowns, simplifying code comprehension, and generating documentation templates, CodeCompanion aims to free up developers’ time and cognitive load, allowing them to focus on the creative and problem-solving aspects of programming. It's also considered a valuable learning tool, helping users explore and understand diverse codebases more easily.

## Features

CodeCompanion offers the following key functionalities:

* **Code Analysis and Explanation:** Provide a code snippet and the tool will generate a natural-language explanation of how the code works, its purpose, and its core logic. Ideal for understanding complex or unfamiliar code.
* **Automatic Documentation:** Provide a code file, and the tool will analyze functions, methods, and classes to automatically generate docstrings based on their parameters, return types (if clear), and perceived functionality. This significantly speeds up the documentation process.
* **README Generation:** Provide a code snippet, and the tool will generate a markdown document summarizing and explaining your code's functionality.

## How It Works

CodeCompanion uses Large Language Models (LLMs) to process your requests. Depending on the selected feature:

1. **Task Breakdown:** The LLM analyzes the task description and generates a structured list of sub-tasks based on its understanding of software development.
2. **Code Analysis:** The LLM reads and interprets the provided code, explaining its structure, logic, and purpose in natural language.
3. **Automatic Documentation:** The LLM examines the signature and content of functions, methods, and classes in the code to infer behavior and generate appropriate docstrings, following common conventions (such as Python’s PEP 257).

The tool interacts with an LLM provider (OpenRouter) through an API key, which must be configured by the user.

## Getting Started

To get started with CodeCompanion, follow these steps:

### Prerequisites

Make sure the following tools are installed on your system:

* `git` (to clone the repository)
* Python 3.8 or higher
* `pip` (Python package manager)

### Cloning the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/Alejandro-Bravo2/proyectoDigitalizacion.git
cd proyectoDigitalizacion
```

### Configure the Environment Variable

Inside the project, you will find a file named `.env` which contains the variable **OPENROUTER\_API\_KEY**. You need to assign an API key generated from OpenRouter to this variable. This is necessary for the program to send requests to OpenRouter's models.

By default, the tool is set to use the **meta-llama/llama-3-8b-instruct** model. If you need a different model (e.g., one with a larger context window), you just need to replace the model name with one available in OpenRouter.

### How to Generate an API Key from OpenRouter

This video will walk you through the process step by step:

[https://www.youtube.com/watch?v=-X9DVzzxpAA](https://www.youtube.com/watch?v=-X9DVzzxpAA)

### How to Choose a Model from OpenRouter

This video explains how to choose a model and what factors to consider:

[https://www.youtube.com/watch?v=vKWz4zdxrvU](https://www.youtube.com/watch?v=vKWz4zdxrvU)

### How to Activate the Virtual Environment (ENV)

To activate the virtual environment, run the following command:

#### Linux & macOS

```bash
source venv/bin/activate
```

#### Windows

```cmd
.venv\Scripts\activate.bat
```

### How to Install Dependencies

To install the necessary dependencies for running the program, run the following command:

```bash
pip3 install -r requeriments.txt
```

**IMPORTANT**
If you're running the program on macOS, you’ll need to install a version of the `urllib3` library lower than 2.0.0.
Command to install a compatible version for macOS:

```bash
pip3 install urllib3==1.26.20
```

### Execution

Once all the steps above are completed, you can run the program either with your IDE or through the command line using:

```bash
python3 main.py
```

### Demo Video of the Program in Action

[Demo video of the program](https://drive.google.com/file/d/1wa9vS7wJ_VF_dX1PMqxVRT9pBJbFwuXk/view?usp=sharing)

