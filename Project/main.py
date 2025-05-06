import gradio as gr
import json
import os
import ast
import re
import openai
from dotenv import load_dotenv
from openai import OpenAI


# --- Configuration ---
# Loads environment variables from a .env file
load_dotenv()

# Loads the OpenRouter API key and model name from environment variables
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL_OPENROUTER = os.environ.get("MODEL_OPENROUTER")

# Checks if the API key is configured and shows an error if not
if not OPENROUTER_API_KEY:
    print("Error: The OPENROUTER_API_KEY environment variable is not configured.")
    print("Please set your OpenRouter API key (sk-or-...) before running.")
    print("Example: export OPENROUTER_API_KEY='sk-or-YOUR_KEY'")
# Checks if the model name is configured and shows a warning if not
if not MODEL_OPENROUTER:
    print("Warning: The MODEL_OPENROUTER environment variable is not configured. Using default model 'meta-llama/llama-3-8b-instruct'.")


# --- AI Interaction Class ---
class DeepSeekDocumenter:
    """
    Handles interactions with the AI model via OpenRouter for code documentation and README generation.
    Streaming is disabled in this version.
    """
    def __init__(self):
        """
        Initializes the OpenAI client for the OpenRouter API.
        Checks for the existence of the API key before initializing the client.
        """
        self.client = None
        # Add an error message if the key is missing from the start
        if not OPENROUTER_API_KEY:
             print("Error: The OPENROUTER_API_KEY environment variable is not configured.")
             print("Please set your OpenRouter API key (sk-or-...) before running.")
             print("Example: export OPENROUTER_API_KEY='sk-or-YOUR_KEY'")
             # Do not initialize the client if the key is missing
             return

        try:
            # Use the openai module directly as a client with explicit base_url and api_key
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY,
            )
            # Optional: verify if the connection is valid (might add latency at startup)
            # print("OpenAI/OpenRouter client initialized successfully.")
            # self.client.models.list() # This would raise an exception if the key is incorrect

        except Exception as e:
            print(f"Error initializing OpenAI (OpenRouter): {e}")
            # Ensure the client is None if initialization fails
            self.client = None


    # --- MODIFIED: No longer a generator (removed yield and stream=True) ---
    def generate_documentation(self, user_input: str, code: str) -> str:
        """
        Generates code documentation by adding comments/docstrings to functions,
        methods, and classes. Returns the complete response at once.

        Args:
            user_input (str): Additional description or context provided by the user.
            code (str): The source code to document.

        Returns:
            str: The complete documented code or an error message.
        """
        # Check if the client was successfully initialized
        if not self.client:
             # Return the error message directly
             return "Error: OpenRouter API client not initialized. Please check your API key."

        # Define the maximum context window size (can vary by model)
        max_length = 12000
        # Simple truncation of the code if it exceeds the maximum size
        truncated_code = code[:max_length]

        # Basic language detection for hints in the prompt
        language_hint = ""
        if re.search(r'^\s*(?:def|async\s+def)\s+\w+\s*\(', code, re.MULTILINE) or re.search(r'^\s*class\s+\w+:', code, re.MULTILINE):
             language_hint = "The code is Python. Use Python docstrings (`\"\"\"Docstring\"\"\"`)."
        elif re.search(r'^\s*(?:fun|suspend\s+fun)\s+\w+\s*\(', code, re.MULTILINE) or re.search(r'^\s*class\s+\w+\s*(?:\(|{)', code, re.MULTILINE):
             language_hint = "The code is Kotlin. Use KDoc (`/** KDoc */`)."
        elif re.search(r'^\s*function\s+\w+\s*\(', code, re.MULTILINE) or re.search(r'^\s*class\s+\w+\s*{', code, re.MULTILINE) or re.search(r'^\s*const\s+\w+\s*=\s*\(?\s*\w*\s*\)?\s*=>', code, re.MULTILINE):
             language_hint = "The code is JavaScript. Use JSDoc comments (`/** JSDoc */`) or line comments (`//`)."
        elif re.search(r'^\s*(?:public|private|protected)?\s*class\s+\w+\s*{', code, re.MULTILINE) or re.search(r'^\s*(?:public|private|protected)?\s*(?:static)?\s*\w+\s+\w+\s*\(', code, re.MULTILINE):
             language_hint = "The code is Java. Use Javadoc (`/** Javadoc */`) or line comments (`//`)."
        elif re.search(r'^\s*(?:func|class)\s+\w+', code, re.MULTILINE):
             language_hint = "The code is Swift. Use documentation comments (`///` or `/** */`)."
        elif re.search(r'^\s*(?:class|struct)\s+\w+', code, re.MULTILINE) or re.search(r'^\s*\w+\s+\w+::\w+\s*\(', code, re.MULTILINE) or re.search(r'^\s*(?:void|int|string|bool|float|double)\s+\w+\s*\(', code, re.MULTILINE):
             language_hint = "The code is C++ or C#. Use documentation comments (`///` or `/** */`) or line comments (`//`)."
        # Add more language hints if needed

        # Build the prompt for the AI model
        prompt = (
            f"You are a code documentation generator. "
            f"Your task is to analyze the following code and return ONLY the original code "
            f"with clear and detailed documentation docstrings/comments added to each function, method, and class."
            f"Describe the purpose, parameters (including type if evident or assumed), and return value."
            f"Ensure the documentation is well-formatted according to the detected language conventions."
            f"DO NOT include explanatory text, introductions, conclusions, or anything that is not the documented code."
            f"{language_hint}\n\n"
            f"Code to document:\n```\n{truncated_code}\n```\n"
            f"Additional user description (if applicable): {user_input}\n\n"
            "Return only the documented code:"
        )
        # Define the messages for the conversation with the model
        messages = [
            {"role": "system", "content": "You are a helpful programming assistant that only returns documented code."},
            {"role": "user", "content": prompt}
        ]

        try:
            # Use the model specified in the environment variable, or a default
            model_to_use = MODEL_OPENROUTER if MODEL_OPENROUTER else "meta-llama/llama-3-8b-instruct"
            completion = self.client.chat.completions.create(
                model=model_to_use, # Model used
                messages=messages,
                temperature=0.2, # Adjust creativity (0.2 is low, focused on precision)
                max_tokens=12000, # Request max tokens up to context window
                # stream=False # Removed or set to False
            )

            # --- MODIFIED: Get the full content directly ---
            # Check for valid response structure
            if not completion or not hasattr(completion, 'choices') or not completion.choices:
                 return "OpenRouter API did not return a valid response (no choices). Please try again or check the model."
            if not hasattr(completion.choices[0], 'message') or not completion.choices[0].message:
                 return "OpenRouter API returned an incomplete response (no message). Please try again or check the model."
            if not hasattr(completion.choices[0].message, 'content') or not completion.choices[0].message.content:
                 return "OpenRouter API returned an empty response (no content). Please try again or check the model."

            # Get the complete content string
            documented_code = completion.choices[0].message.content

            # Note: The post-generation filtering and translation methods
            # (_filter_documentation, _translate_documentation) can be applied here
            # if needed, as we have the full content. Keeping them commented for now
            # as the prompt aims for direct output.
            # processed_code = self._translate_documentation(self._filter_documentation(documented_code))
            processed_code = documented_code # Use the raw content

            # Check for empty processed content
            if not processed_code or not processed_code.strip():
                 return "OpenRouter API did not return documented code after processing. Please try with different code or check the model."

            return processed_code # Return the complete string

        except openai.APIError as e:
            # Capture and return specific API errors
            print(f"OpenAI/OpenRouter API Error: {e}")
            return f"Error generating documentation (API Error): {e}" # Return the error
        except Exception as e:
            # Capture and return other unexpected errors
            print(f"Unexpected error: {e}")
            return f"Unexpected error during generation: {e}" # Return the error


    # --- MODIFIED: No longer a generator (removed yield and stream=True) ---
    def generate_readme_content(self, user_description: str, code: str, existing_readme: str = "") -> str:
         """
         Generates the content of a README.md file based on the code and a description.
         Returns the complete response at once.

         Args:
             user_description (str): Additional description or instructions from the user for the README.
             code (str): The source code of the project for which to generate the README.
             existing_readme (str, optional): Content of an existing README.md to base upon.
                                              Defaults to an empty string.

         Returns:
             str: The complete README.md content or an error message.
         """
         # Check if the client was successfully initialized
         if not self.client:
             return "Error: OpenRouter API client not initialized. Please check your API key." # Return the error


         # Define the maximum context window size and truncate the code
         max_length = 12000
         # Use half the context for the code, the other half for the prompt/readme
         truncated_code = code[:max_length // 2]

         # Build the prompt for the AI model to generate the README
         prompt = (
             f"You are an expert README.md file generator in Markdown format."
             f"Your task is to create a complete and well-structured README.md file for a software project."
             f"Include standard sections like Title, Project Description, Features, Installation, Usage, Code Structure, Contribution, License, etc."
             f"Base your content on the following project code and existing README content (if provided)."
             f"Ensure the README is clear, concise, and easy for other developers to understand."
             f"Pay close attention to any additional instructions or description provided by the user."
             f"\n\n--- Existing README.md Content (if any) ---\n{existing_readme}\n"
             f"\n\n--- Project Code ---\n```\n{truncated_code}\n```\n"
             f"\n\n--- Additional User Instructions ---\n{user_description}\n"
             f"\n\n--- New README.md Content (in Markdown) ---"
             f"\nReturn ONLY the complete README.md content in Markdown format, with no additional text before or after."
         )

         # Define the messages for the conversation with the model
         messages = [
             {"role": "system", "content": "You are an expert README.md file generator in Markdown format and only return Markdown content."},
             {"role": "user", "content": prompt}
         ]

         try:
             # Use the model specified in the environment variable, or a default
             model_to_use = MODEL_OPENROUTER if MODEL_OPENROUTER else "meta-llama/llama-3-8b-instruct"
             completion = self.client.chat.completions.create(
                 model=model_to_use, # Suitable model (consistent with documentation for testing)
                 messages=messages,
                 temperature=0.7, # Slightly higher temperature for creativity in README
                 max_tokens=12000, # Request max tokens
                 # stream=False # Removed or set to False
             )

             # --- MODIFIED: Get the full content directly ---
             # Check for valid response structure
             if not completion or not hasattr(completion, 'choices') or not completion.choices:
                  return "OpenRouter API did not return a valid response (no choices). Please try again or check the model."
             if not hasattr(completion.choices[0], 'message') or not completion.choices[0].message:
                  return "OpenRouter API returned an incomplete response (no message). Please try again or check the model."
             if not hasattr(completion.choices[0].message, 'content') or not completion.choices[0].message.content:
                  return "OpenRouter API returned an empty response (no content). Please try again or check the model."

             readme_content = completion.choices[0].message.content

             # Simple filtering for Markdown code blocks if the AI mistakenly wraps it
             readme_content = readme_content.strip()
             if readme_content.startswith("```markdown"):
                  readme_content = readme_content[len("```markdown"):].strip()
             if readme_content.endswith("```"):
                  readme_content = readme_content[:-len("```")].strip()

             # Check for empty processed content
             if not readme_content or not readme_content.strip():
                  return "OpenRouter API did not return content for the README. Please try with a different description or check the model."


             return readme_content # Return the complete string

         except openai.APIError as e:
             # Capture and return specific API errors
             print(f"OpenAI/OpenRouter API Error generating README: {e}")
             return f"Error generating README (API Error): {e}" # Return the error
         except Exception as e:
             # Capture and return other unexpected errors
             print(f"Unexpected error generating README: {e}")
             return f"Unexpected error during README generation: {e}" # Return the error


    def _filter_documentation(self, doc):
        """
        Filters unnecessary additional information that the model might include.
        (This method is not currently used in the streaming flow).
        """
        # The prompt asks for only code, so less filtering is expected to be needed.
        # A simple strip might suffice, or look for common AI conversational filler.
        # Also remove markdown code block fences if present
        doc = doc.strip()
        if doc.startswith("```") and doc.endswith("```"):
             # Remove language specifier like ```python
             doc = re.sub(r"^```[a-zA-Z]*\n", "", doc)
             doc = re.sub(r"\n```$", "", doc)
        return doc.strip()


    def _translate_documentation(self, doc):
        """
        Translates the documentation to Spanish if necessary.
        (This method is not currently used in the streaming flow).
        """
        # If the prompt was in Spanish and the target is Spanish, this function might just return the input.
        # Since we are now prompting in English, this function effectively just returns the input.
        return doc

# --- Code Analysis Classes (Do not need changes for streaming) ---

class PythonCodeStructureAnalyzer(ast.NodeVisitor):
    """
    Analyzes the structure of Python code to extract information about classes,
    functions, and their basic details (names, arguments, docstrings).
    """
    def __init__(self):
        """Initializes lists to store class and function information."""
        self.classes = []
        self.functions = []

    def visit_ClassDef(self, node):
        """
        Visits a ClassDef node (class definition) in the Python AST
        and extracts the class name, its docstring, and the methods it contains.
        """
        class_info = {
            'name': node.name,
            'methods': []
        }
        # Get the class docstring
        class_docstring = ast.get_docstring(node)
        if class_docstring:
             class_info['description'] = class_docstring.strip()
        else:
             class_info['description'] = "No description provided."

        # Iterate over the elements within the class body to find methods
        for body_item in node.body:
            if isinstance(body_item, ast.FunctionDef):
                method_info = {
                    'name': body_item.name,
                    # Extract the names of the method arguments
                    'args': [arg.arg for arg in body_item.args.args],
                    # Get the method docstring directly
                    'description': ast.get_docstring(body_item) or "No description provided."
                }
                class_info['methods'].append(method_info)
        # Add the class information to the list of found classes
        self.classes.append(class_info)
        # Continue visiting child nodes (in case of nested classes, although not fully handled)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        """
        Visits a FunctionDef node (function definition) in the Python AST
        and extracts the function name, its arguments, and its docstring.
        Only applies to top-level functions (not class methods).
        """
        func_info = {
            'name': node.name,
            # Extract the names of the function arguments
            'args': [arg.arg for arg in node.args.args],
            # Get the function docstring directly
            'description': ast.get_docstring(node) or "No description provided."
        }
        # Add the function information to the list of found functions
        self.functions.append(func_info)
        # Continue visiting child nodes
        self.generic_visit(node)

    def get_report(self):
        """
        Generates a formatted plain text report of the analyzed Python code structure,
        listing classes, their methods, and top-level functions with their descriptions.
        """
        report = "Code Structure Analysis (Python):\n\n"
        if self.classes:
            report += "Classes\n" + "="*50 + "\n"
            for cls in self.classes:
                report += f"Class: {cls['name']}\n"
                # Add class docstring if available, truncating if very long
                if cls.get('description') and cls['description'] != "No description provided.":
                     report += f"  Description: {cls['description'][:200]}...\n"
                if cls['methods']:
                    report += "  Methods:\n"
                    for method in cls['methods']:
                        args = ", ".join(method['args'])
                        report += f"    - {method['name']}({args})\n"
                        # Optionally add method description here for the report text, truncating
                        if method['description'] and method['description'] != "No description provided.":
                           report += f"      Description: {method['description'][:200]}...\n"
                report += "\n"
        if self.functions:
            report += "Top-Level Functions\n" + "="*50 + "\n"
            for func in self.functions:
                args = ", ".join(func['args'])
                report += f"  - {func['name']}({args})\n"
                # Optionally add function description here, truncating
                if func['description'] and func['description'] != "No description provided.":
                   report += f"    Description: {func['description'][:200]}...\n"

        # Message if no classes or functions were found
        if not self.classes and not self.functions:
            report += "No classes or functions found in the provided code."
        return report


class KotlinCodeStructureAnalyzer:
    """
    Analyzes the structure of Kotlin code (using basic heuristic analysis
    based on regex) to extract information about classes and functions.
    """
    def __init__(self):
        """Initializes lists to store class and function information."""
        self.classes = []
        self.functions = []

    def parse(self, code):
        """
        Parses Kotlin code to extract basic class and function information.
        Note: This is a simplified parser using regex and heuristics, not a full AST parser.

        Args:
            code (str): The Kotlin source code to analyze.
        """
        # Regex patterns to identify classes, functions, and KDoc comments
        class_pattern = re.compile(r'^\s*class\s+(\w+)', re.MULTILINE)
        method_pattern = re.compile(r'^\s*fun\s+(\w+)\s*\(([^)]*)\)', re.MULTILINE)
        # This description pattern looks for multi-line KDoc comments right before fun/class
        description_pattern = re.compile(r'/\*\*(.*?)\*/\s*$', re.DOTALL | re.MULTILINE)

        current_class_name = None
        lines = code.split('\n')
        class_start_line = -1

        # Reset lists for a new analysis
        self.classes = []
        self.functions = []

        i = 0
        while i < len(lines):
            line = lines[i]

            # Check for class definition
            class_match = class_pattern.search(line)
            if class_match:
                current_class_name = class_match.group(1)
                class_start_line = i
                class_info = {'name': current_class_name, 'methods': []}
                # Find description *before* the class line
                description = "No description provided."
                # Look back a few lines for a KDoc comment
                for j in range(i - 1, max(-1, i - 5), -1): # Check up to 5 lines back
                    desc_match = description_pattern.search(lines[j])
                    if desc_match:
                        description = desc_match.group(1).strip()
                        break
                class_info['description'] = description
                self.classes.append(class_info)
                # Methods will be added as found within the class block

            # Check for method definition
            method_match = method_pattern.search(line)
            if method_match:
                method_name = method_match.group(1)
                # Simple argument extraction (argument name before ':' or '=')
                args_str = method_match.group(2)
                args = [arg.split(':')[0].strip().split('=')[0].strip() for arg in args_str.split(',') if arg.strip()] if args_str.strip() else []

                # Find description *before* the method line
                description = "No description provided."
                # Look back a few lines for a KDoc comment
                for j in range(i - 1, max(-1, i - 5), -1): # Check up to 5 lines back
                    desc_match = description_pattern.search(lines[j])
                    if desc_match:
                        description = desc_match.group(1).strip()
                        break

                method_info = {
                    'name': method_name,
                    'args': args,
                    'description': description
                }
                if current_class_name:
                    # If inside a class, add the method to that class
                    # This assumes methods are listed directly under the class definition
                    for cls in self.classes:
                        if cls['name'] == current_class_name:
                            cls['methods'].append(method_info)
                            break # Class found
                else:
                    # If not inside a class, it's a top-level function
                    self.functions.append(method_info)

            # Simple logic to detect the end of a class block (might need refinement)
            # Assuming simple bracket structure { ... } and counting the balance
            if current_class_name and line.strip() == '}':
                 brace_balance = 0
                 # Count braces from the start of the class to the current line
                 for k in range(class_start_line, i + 1):
                      brace_balance += lines[k].count('{')
                      brace_balance -= lines[k].count('}')
                 # If the balance is 0, we assume it's the end of the current class
                 if brace_balance == 0:
                      current_class_name = None
                      class_start_line = -1

            i += 1 # Move to the next line


    def get_report(self):
        """
        Generates a formatted plain text report of the analyzed Kotlin code structure,
        listing classes, their methods, and top-level functions with their descriptions.
        """
        report = "Code Structure Analysis (Kotlin):\n\n"
        if self.classes:
            report += "Classes\n" + "="*50 + "\n"
            for cls in self.classes:
                report += f"Class: {cls['name']}\n"
                # Add class description if available, truncating
                if cls.get('description') and cls['description'] != "No description provided.":
                     report += f"  Description: {cls['description'][:200]}...\n" # Truncate
                if cls['methods']:
                    report += "  Methods:\n"
                    for method in cls['methods']:
                        args = ", ".join(method['args'])
                        report += f"    - {method['name']}({args})\n"
                        # Add method description if available, truncating
                        if method['description'] and method['description'] != "No description provided.":
                             report += f"      Description: {method['description'][:200]}...\n" # Truncate
                report += "\n"
        if self.functions:
            report += "Top-Level Functions\n" + "="*50 + "\n"
            for func in self.functions:
                args = ", ".join(func['args'])
                report += f"  - {func['name']}({args})\n"
                # Add function description if available, truncating
                if func['description'] and func['description'] != "No description provided.":
                     report += f"    Description: {func['description'][:200]}...\n" # Truncate

        # Message if no classes or functions were found
        if not self.classes and not self.functions:
            report += "No classes or functions found in the provided code."
        return report


# --- Gradio Interface Functions ---

# Initialize the AI documenter instance
ai_documenter_instance = DeepSeekDocumenter()

# --- MODIFIED: No longer a generator (removed yield from) ---
def gradio_document_code(description: str, code: str) -> str:
    """
    Gradio function to handle the code documentation task.
    Calls the documentation method from the DeepSeekDocumenter class and returns the complete output.

    Args:
        description (str): Additional description provided by the user.
        code (str): The code to document.

    Returns:
        str: The complete documented code or an error message.
    """
    if not code:
        return "Please enter the code to document."

    # Call the documentation method and return its complete output
    return ai_documenter_instance.generate_documentation(description, code)


def gradio_analyze_code(code: str, language: str) -> str:
    """
    Gradio function to handle the code analysis task.
    Uses the structure analysis classes based on the selected language.

    Args:
        code (str): The code to analyze.
        language (str): The code language (e.g., "Python", "Kotlin").

    Returns:
        str: A report of the code structure or an error message.
    """
    if not code:
        return "Please enter the code to analyze."
    if not language:
        return "Please select the code language."

    try:
        if language.lower() == "python":
            analyzer = PythonCodeStructureAnalyzer()
            tree = ast.parse(code) # Parse the Python code into an Abstract Syntax Tree (AST)
            analyzer.visit(tree) # Visit the AST to extract information
            report = analyzer.get_report() # Get the formatted report
        elif language.lower() == "kotlin":
            analyzer = KotlinCodeStructureAnalyzer()
            analyzer.parse(code) # Use the heuristic parser for Kotlin
            report = analyzer.get_report() # Get the formatted report
        else:
            report = f"Error: Language '{language}' not supported for analysis."
        return report
    except Exception as e:
        # Capture errors during analysis (e.g., syntax errors)
        return f"Error during code analysis: {e}"


# --- MODIFIED: No longer a generator (removed yield from) ---
def gradio_generate_readme(description: str, code_for_readme_input: str) -> str:
    """
    Gradio function to handle the README.md generation task.
    Reads 'main.py' or uses the input code, reads an existing README if it exists,
    and calls the README generator from the DeepSeekDocumenter class, returning the complete response.

    Args:
        description (str): Additional description or instructions for the README.
        code_for_readme_input (str): Optional code pasted by the user if not using main.py.

    Returns:
        str: The complete generated README.md or an error message.
    """
    main_py_code = ""
    readme_content = ""

    try:
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        main_py_path = os.path.join(script_dir, "main.py")
        readme_path = os.path.join(script_dir, "README.md")

        # Try to read main.py first
        if os.path.exists(main_py_path):
            with open(main_py_path, "r", encoding="utf-8") as f:
                main_py_code = f.read()
        elif code_for_readme_input:
             # If main.py does not exist, use the code from the text field
             main_py_code = code_for_readme_input
        else:
            # If there is no main.py and no code in the field, return an error
            return "Error: 'main.py' not found and no code provided in the text field."


        # Try to read the existing README.md
        if os.path.exists(readme_path):
            try:
                with open(readme_path, "r", encoding="utf-8") as rf:
                    readme_content = rf.read()
            except Exception as e:
                # You can print a warning message to the console if you want
                print(f"Warning: Could not read existing README ({e}).")
                # Continue without existing README content


    except Exception as e:
        # Capture errors during file reading
        return f"Error reading files for README generation: {e}"

    # Call the DeepSeekDocumenter class generator and return its complete output
    return ai_documenter_instance.generate_readme_content(
        user_description=description,
        code=main_py_code, # Pass the read code (or from input)
        existing_readme=readme_content # Pass the existing README content
    )


# --- Gradio Interface Definition ---

# Create a main block for the Gradio interface
with gr.Blocks() as demo:
    # Main application title
    gr.Markdown("# AI Code Assistant with OpenRouter")
    # Descriptive subtitle
    gr.Markdown("Document, analyze code, and generate READMEs using models via OpenRouter.ai.")

    # Tab for Documenting Code
    with gr.Tab("Document Code"): # Tab title
        gr.Markdown("## Document Code") # Section title
        # Textbox for additional description
        doc_description_input = gr.Textbox(
            label="Additional Description (Optional)", # Label
            placeholder="e.g., This code manages the user database.", # Placeholder
            lines=2
        )
        # Textbox to paste source code
        code_input_doc = gr.Textbox(
            label="Paste Your Code Here", # Label
            lines=20,
            placeholder="class MyClass:\n    def my_method(self):...", # Placeholder
            show_copy_button=True # Show copy button
        )
        # Button to start documentation
        document_button = gr.Button("Document Code") # Button text
        # Textbox to display documented code (non-interactive)
        documented_code_output = gr.Textbox(
            label="Documented Code", # Label
            lines=20,
            interactive=False, # Not editable by the user
            show_copy_button=True # Show copy button
        )
        # Configure the button action: calls the gradio_document_code function
        # This function now returns a single string, so Gradio will display it when ready
        document_button.click(
            fn=gradio_document_code,
            inputs=[doc_description_input, code_input_doc],
            outputs=documented_code_output
        )

    # Tab for Analyzing Code
    with gr.Tab("Analyze Code"): # Tab title
        gr.Markdown("## Analyze Code") # Section title
        # Textbox to paste code to analyze
        code_input_analyze = gr.Textbox(
            label="Paste Your Code Here", # Label
            lines=20,
            placeholder="class MyClass:\n    fun myMethod() {...}", # Placeholder
             show_copy_button=True # Show copy button
        )
        # Radio buttons to select the code language
        language_select = gr.Radio(
            ["Python", "Kotlin", "JavaScript", "Java", "Swift", "C++ / C#"], # Language options (kept as names)
            label="Code Language", # Label
            value="Python" # Default selected value
        )
        # Button to start analysis
        analyze_button = gr.Button("Analyze Code") # Button text
        # Textbox to display the analysis report (non-interactive)
        analysis_output = gr.Textbox(
            label="Code Analysis Report", # Label
            lines=20,
            interactive=False, # Not editable by the user
            show_copy_button=True # Show copy button
        )
        # Configure the button action: calls the gradio_analyze_code function
        # This function does not use streaming, no changes needed here.
        analyze_button.click(
            fn=gradio_analyze_code,
            inputs=[code_input_analyze, language_select],
            outputs=analysis_output
        )

    # Tab for Generating README
    with gr.Tab("Generate README"): # Tab title
        gr.Markdown("## Generate README") # Section title
        # Textbox for additional description/instructions for the README
        readme_description_input = gr.Textbox(
            label="Additional Description / Instructions for README (Optional)", # Label
            placeholder="e.g., Focus on the 'Installation' section.", # Placeholder
            lines=2
        )
        # Optional textbox to paste code if main.py is not used
        code_input_readme = gr.Textbox(
            label="Paste Code Here (if not using main.py)", # Label
            lines=10,
            placeholder="Optional: Paste code if 'main.py' doesn't exist or you want to include code from another file.", # Placeholder
            show_copy_button=True # Show copy button
        )
        # Button to start README generation
        generate_readme_button = gr.Button("Generate README") # Button text
        # Textbox to display the generated README (non-interactive)
        readme_output = gr.Textbox(
            label="Generated README (Markdown Format)", # Label
            lines=20,
            interactive=False, # Not editable by the user
            show_copy_button=True # Show copy button
        )
        # Configure the button action: calls the gradio_generate_readme function
        # This function now returns a single string, so Gradio will display it when ready
        generate_readme_button.click(
            fn=gradio_generate_readme,
            inputs=[readme_description_input, code_input_readme],
            outputs=readme_output
        )

# --- Launch the Gradio App ---
# When the script is executed directly
if __name__ == "__main__":
    # Launches the Gradio interface. share=False to not share publicly.
    demo.launch(share=False)
