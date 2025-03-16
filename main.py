import tkinter as tk
from tkinter import simpledialog, ttk, scrolledtext
import tkinter.messagebox as messagebox 
import tkinter.font as tkFont
from pynput import keyboard
import json
import os
import threading
import ast
import re
import openai 



# Clase para gestionar las solicitudes a la IA
class DeepSeekDocumenter:
    def __init__(self):
        try:
            openai.api_base = "http://localhost:1234/v1"
            openai.api_key = "llama-3.2-1b-instruct"
            self.client = openai
        except Exception as e:
            raise ValueError(f"Error al inicializar OpenAI: {e}")

    def generate_documentation(self, user_input, code):
        """
        Genera documentación del código agregando comentarios claros y concisos en cada función, clase y método.
        La respuesta contendrá únicamente el código documentado, sin explicaciones adicionales.
        """
        max_length = 12000
        truncated_code = code if len(code) <= max_length else code[:max_length] + "\n... [truncated]"
        language_note = ""
        if "package " in code or "class " in code:
            language_note = "Nota: El código está escrito en Kotlin. "
        prompt = (
            f"{language_note}"
            """
            “Eres un generador de documentación para código Python, kotlin o cualquier lenguaje. Tu tarea es analizar el siguiente código y devolver el mismo código sin interrupciones, pero agregando docstrings a cada función, método y clase. Asegúrate de documentar de manera clara y detallada el propósito de cada elemento, incluyendo la descripción de cada parámetro (con su tipo correspondiente) y, en su caso, el valor de retorno. No incluyas comentarios adicionales fuera de los docstrings. Usa el siguiente ejemplo como guía:

python
Copiar
Editar
def _filtrar_documentacion(self, doc):
    <<DOC>>
    Filtra información adicional innecesaria del documento de documentación.

    Args:
        doc (str): La documentación a filtrar.

    Returns:
        str: La documentación filtrada, sin espacios al inicio o al final.
    <<DOC>>
    return doc.strip()

def _traducir_documentacion(self, doc):
    <<DOC>>
    Traduce la documentación al español en caso de ser necesario.

    Se asume que la documentación ya viene en español.

    Args:
        doc (str): La documentación a traducir.

    Returns:
        str: La documentación traducida.
    <<DOC>>
    return doc

def _mostrar_resultado(title, content, parent=None):
    <<DOC>>
    Muestra el contenido proporcionado en una ventana nueva.

    Se crea una ventana hija (Toplevel) con un área de texto desplazable que contiene el contenido.

    Args:
        title (str): Título de la ventana.
        content (str): Contenido a mostrar en la ventana.
        parent (tk.Widget, opcional): Widget padre para la ventana. Si no se especifica, se utiliza el widget raíz predeterminado.
    <<DOC>>
    if parent is None:
        parent = tk._get_default_root()
    result_window = tk.Toplevel(parent)
    result_window.title(title)
    st = scrolledtext.ScrolledText(result_window, width=80, height=30)
    st.pack(expand=True, fill='both')
    st.insert(tk.END, content)
Por favor, documenta todo el código que se te proporcione siguiendo exactamente este formato, sin interrupciones ni explicaciones adicionales, solo el código modificado con los docstrings correspondientes.”
            """
            f"{truncated_code}"
            "\n\nDescripción adicional: " + user_input
        )
        messages = [
            {"role": "system", "content": "Eres un asistente útil."},
            {"role": "user", "content": prompt}
        ]
        try:
            completion = self.client.ChatCompletion.create(
                model="deepseek-r1-llama-8b",
                messages=messages,
                temperature=0.7,
                max_tokens=12000
            )
            documented_code = completion.choices[0].message.content
            return self._traducir_documentacion(self._filtrar_documentacion(documented_code))
        except Exception as e:
            messagebox.showerror("Error", f"Error al generar la documentación: {e}")
            return "Error en la generación de la documentación."

    def _filtrar_documentacion(self, doc):
        """Filtra información adicional innecesaria."""
        return doc.strip()

    def _traducir_documentacion(self, doc):
        """Traduce la documentación al español en caso de ser necesario (asumimos que ya viene en español)."""
        return doc

# Función para mostrar resultados en una única ventana
def _mostrar_resultado(title, content, parent=None):
    if parent is None:
        parent = tk._get_default_root()
    result_window = tk.Toplevel(parent)
    result_window.title(title)
    st = scrolledtext.ScrolledText(result_window, width=80, height=30)
    st.pack(expand=True, fill='both')
    st.insert(tk.END, content)

# Función para analizar código
def analyze_code():
    code = simpledialog.askstring("Código a Analizar", "Introduce el código a analizar:")
    if not code:
        messagebox.showinfo("Análisis cancelado", "No se proporcionó código para analizar.")
        return
    ds = DeepSeekDocumenter()
    prompt = "Analiza el siguiente código y explica su funcionamiento:"
    result = ds.generate_documentation(prompt, code)
    _mostrar_resultado("Análisis de Código", result)

# Función para documentar código
def document_code_button():
    code = simpledialog.askstring("Código a Documentar", "Introduce el código a documentar:")
    if not code:
        messagebox.showinfo("Documentación cancelada", "No se proporcionó código para documentar.")
        return
    ds = DeepSeekDocumenter()
    prompt = "Documenta el siguiente código, añadiendo comentarios claros a funciones, clases y métodos:"
    result = ds.generate_documentation(prompt, code)
    _mostrar_resultado("Documentación del Código", result)

# Función para generar README a partir del archivo main.py y el README existente, si lo hay
def document_readme_button():
    try:
        with open("main.py", "r", encoding="utf-8") as f:
            code = f.read()
    except Exception as e:
        messagebox.showerror("Error al leer archivo", str(e))
        return

    existing_readme = ""
    if os.path.exists("README.md"):
        try:
            with open("README.md", "r", encoding="utf-8") as rf:
                existing_readme = rf.read()
        except Exception as e:
            messagebox.showwarning("Advertencia", f"No se pudo leer el README existente: {e}")

    ds = DeepSeekDocumenter()
    prompt = (
        "Genera una documentación en formato README para este proyecto. "
        "Utiliza el siguiente código y la información existente como referencia:\n\n"
        f"{existing_readme}\n\nCódigo:\n"
    )
    result = ds.generate_documentation(prompt, code)
    _mostrar_resultado("Generar README", result)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECTS_FILE = os.path.join(BASE_DIR, "projects.json")

def load_projects():
    if os.path.exists(PROJECTS_FILE):
        with open(PROJECTS_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)
            # Asegurar que el dato cargado es una lista
            if isinstance(data, list):
                return data
            else:
                return []
    return []

def save_projects(projects):
    with open(PROJECTS_FILE, "w", encoding="utf-8") as file:
        json.dump(projects, file, indent=4, ensure_ascii=False)

def on_activate():
    open_project_window()

def for_canonical(f):
    return lambda k: f(l.canonical(k))

def start_listener():
    with keyboard.GlobalHotKeys({
            '<ctrl>+<alt>+p': on_activate}) as l:
        l.join()

class CustomAskString(tk.Toplevel):
    def __init__(self, parent, title, prompt, initialvalue=''):
        super().__init__(parent)
        self.title(title)
        self.configure(bg="#1a1a2e")
        self.resizable(False, False)
        self.result = None

        # Etiqueta de la pregunta
        tk.Label(self, text=prompt, bg="#1a1a2e", fg="#00ffea", font=("Courier", 12)).pack(padx=10, pady=5)

        # Entrada de texto
        self.entry = tk.Entry(self, bg="#16213e", fg="#00ffea", font=("Courier", 12))
        self.entry.pack(padx=10, pady=5)
        self.entry.insert(0, initialvalue)
        self.entry.focus_set()

        # Frame para los botones
        button_frame = tk.Frame(self, bg="#1a1a2e")
        button_frame.pack(pady=5)

        # Botón OK
        ok_button = tk.Button(button_frame, text="OK", command=self.on_ok, bg="#0f3460", fg="#ffffff",
                              font=("Courier", 10, "bold"), width=10)
        ok_button.pack(side=tk.LEFT, padx=2)

        # Botón Cancel
        cancel_button = tk.Button(button_frame, text="Cancel", command=self.on_cancel, bg="#e94560", fg="#ffffff",
                                  font=("Courier", 10, "bold"), width=10)
        cancel_button.pack(side=tk.LEFT, padx=2)

        # Atajos de teclado
        self.bind("<Return>", lambda event: self.on_ok())
        self.bind("<Escape>", lambda event: self.on_cancel())

        # Modalidad de la ventana
        self.grab_set()
        self.transient(parent)
        parent.wait_window(self)

    def on_ok(self):
        self.result = self.entry.get()
        self.destroy()

    def on_cancel(self):
        self.destroy()

def custom_askstring(title, prompt, parent=None, initialvalue=''):
    dialog = CustomAskString(parent, title, prompt, initialvalue)
    return dialog.result

class CodeInputDialog(tk.Toplevel):
    def __init__(self, parent, title, prompt):
        super().__init__(parent)
        self.title(title)
        self.geometry("600x400")
        self.configure(bg="#1a1a2e")
        self.result = None
        self.resizable(False, False)

        # Etiqueta de la pregunta
        tk.Label(self, text=prompt, bg="#1a1a2e", fg="#00ffea", font=("Courier", 12)).pack(padx=10, pady=10)

        # Área de texto para código
        self.text_area = scrolledtext.ScrolledText(self, bg="#16213e", fg="#00ffea",
                                                   font=("Courier", 12), wrap=tk.WORD, height=15, width=70)
        self.text_area.pack(padx=5, pady=5)
        self.text_area.focus_set()

        # Frame para los botones
        button_frame = tk.Frame(self, bg="#1a1a2e")
        button_frame.pack(pady=5)

        # Botón OK
        ok_button = tk.Button(button_frame, text="OK", command=self.on_ok, bg="#0f3460", fg="#ffffff",
                              font=("Courier", 10, "bold"), width=10)
        ok_button.pack(side=tk.LEFT, padx=2)

        # Botón Cancel
        cancel_button = tk.Button(button_frame, text="Cancel", command=self.on_cancel, bg="#e94560", fg="#ffffff",
                                  font=("Courier", 10, "bold"), width=10)
        cancel_button.pack(side=tk.LEFT, padx=2)

        # Atajos de teclado
        self.bind("<Return>", lambda event: self.on_ok())
        self.bind("<Escape>", lambda event: self.on_cancel())

        # Modalidad de la ventana
        self.grab_set()
        self.transient(parent)
        parent.wait_window(self)

    def on_ok(self):
        self.result = self.text_area.get("1.0", tk.END).strip()
        self.destroy()

    def on_cancel(self):
        self.destroy()

class DocumentDialog(tk.Toplevel):
    def __init__(self, parent, title, prompt):
        super().__init__(parent)
        self.title(title)
        self.geometry("400x200")
        self.configure(bg="#1a1a2e")
        self.result = None
        self.resizable(False, False)

        # Etiqueta de la pregunta
        tk.Label(self, text=prompt, bg="#1a1a2e", fg="#00ffea", font=("Courier", 12)).pack(padx=10, pady=10)

        # Entrada de texto
        self.entry = scrolledtext.ScrolledText(self, bg="#16213e", fg="#00ffea",
                                              font=("Courier", 12), wrap=tk.WORD, height=5, width=40)
        self.entry.pack(padx=10, pady=5)
        self.entry.focus_set()

        # Frame para los botones
        button_frame = tk.Frame(self, bg="#1a1a2e")
        button_frame.pack(pady=5)

        # Botón OK
        ok_button = tk.Button(button_frame, text="OK", command=self.on_ok, bg="#0f3460", fg="#ffffff",
                              font=("Courier", 10, "bold"), width=10)
        ok_button.pack(side=tk.LEFT, padx=2)

        # Botón Cancel
        cancel_button = tk.Button(button_frame, text="Cancel", command=self.on_cancel, bg="#e94560", fg="#ffffff",
                                  font=("Courier", 10, "bold"), width=10)
        cancel_button.pack(side=tk.LEFT, padx=2)

        # Atajos de teclado
        self.bind("<Return>", lambda event: self.on_ok())
        self.bind("<Escape>", lambda event: self.on_cancel())

        # Modalidad de la ventana
        self.grab_set()
        self.transient(parent)
        parent.wait_window(self)

    def on_ok(self):
        self.result = self.entry.get("1.0", tk.END).strip()
        self.destroy()

    def on_cancel(self):
        self.destroy()

class DocumentedCodeWindow(tk.Toplevel):
    def __init__(self, parent, documented_code):
        super().__init__(parent)
        self.title("Código Documentado")
        self.geometry("800x600")
        self.configure(bg="#1a1a2e")
        self.resizable(True, True)

        # Área de texto para mostrar el código documentado
        self.text_area = scrolledtext.ScrolledText(self, bg="#16213e", fg="#00ffea",
                                                   font=("Courier", 12), wrap=tk.WORD)
        self.text_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.text_area.insert(tk.END, documented_code)
        self.text_area.configure(state='disabled')

        # Botón para copiar el código
        copy_button = tk.Button(self, text="Copiar Código", command=lambda: self.copy_to_clipboard(documented_code),
                                bg="#0f3460", fg="#ffffff", font=("Courier", 10, "bold"), width=15)
        copy_button.pack(pady=5)

    def copy_to_clipboard(self, text):
        self.clipboard_clear()
        self.clipboard_append(text)
        messagebox.showinfo("Copiado", "El código documentado ha sido copiado al portapapeles.")

class CodeAnalyzer(tk.Toplevel):
    def __init__(self, parent, code, language):
        super().__init__(parent)
        self.title("Código Analizado")
        self.geometry("800x600")
        self.configure(bg="#1a1a2e")
        self.resizable(True, True)
        self.language = language.lower()

        # Diccionario para almacenar descripciones de métodos
        self.method_descriptions = {}

        # Frame para el análisis y detalles
        self.analysis_frame = tk.Frame(self, bg="#1a1a2e")
        self.analysis_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Frame para el reporte del análisis
        report_frame = tk.Frame(self.analysis_frame, bg="#1a1a2e")
        report_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,5))

        # Lista para mostrar clases y métodos
        self.tree = ttk.Treeview(report_frame, columns=("Type", "Name"), show='headings')
        self.tree.heading("Type", text="Tipo")
        self.tree.heading("Name", text="Nombre")
        self.tree.column("Type", width=100, anchor='center')
        self.tree.column("Name", width=200, anchor='w')
        self.tree.pack(fill=tk.BOTH, expand=True)

        # Bind para manejar la selección en el Treeview
        self.tree.bind("<<TreeviewSelect>>", self.display_method_description)

        # Frame para los detalles del método
        details_frame = tk.Frame(self.analysis_frame, bg="#1a1a2e")
        details_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5,0))

        details_label = tk.Label(details_frame, text="Detalles del Método:", bg="#1a1a2e", fg="#00ffea",
                                 font=("Courier", 12, "bold"))
        details_label.pack(anchor='w')

        self.details_text = scrolledtext.ScrolledText(details_frame, bg="#16213e", fg="#00ffea",
                                                    font=("Courier", 12), wrap=tk.WORD, height=10, state='disabled')
        self.details_text.pack(fill=tk.BOTH, expand=True)

        # Botón para cerrar la ventana de análisis
        close_button = tk.Button(self, text="Cerrar", command=self.destroy,
                                 bg="#e94560", fg="#ffffff", font=("Courier", 10, "bold"), width=10)
        close_button.pack(pady=5)

        # Analizar el código y llenar el Treeview
        self.report = self.analyze_code(code)
        self.populate_treeview()

    def analyze_code(self, code):
        try:
            if self.language == "python":
                tree = ast.parse(code)
                analyzer = CodeStructureAnalyzer()
                analyzer.visit(tree)
                self.method_descriptions = analyzer.method_descriptions
                return analyzer.get_report()
            elif self.language == "kotlin":
                analyzer = KotlinCodeStructureAnalyzer()
                analyzer.parse(code)
                self.method_descriptions = analyzer.method_descriptions
                return analyzer.get_report()
            else:
                return f"Lenguaje '{self.language}' no soportado para análisis."
        except Exception as e:
            return f"Error al analizar el código: {e}"

    def populate_treeview(self):
        # Limpiar el Treeview antes de llenarlo
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Insertar clases y métodos
        for cls in self.method_descriptions.get('classes', []):
            class_id = self.tree.insert("", "end", values=("Clase", cls['name']))
            for method in cls['methods']:
                self.tree.insert(class_id, "end", values=("Método", method['name']))

        # Insertar funciones si existen
        for func in self.method_descriptions.get('functions', []):
            self.tree.insert("", "end", values=("Función", func['name']))

    def display_method_description(self, event):
        selected_item = self.tree.selection()
        if not selected_item:
            return
        item = selected_item[0]
        item_type, item_name = self.tree.item(item, 'values')
        
        description = "No hay descripción disponible."
        
        if item_type == "Método":
            for cls in self.method_descriptions['classes']:
                for method in cls['methods']:
                    if method['name'] == item_name:
                        description = method['description']
                        args = method.get('args', [])
                        # Si no hay descripción, la genera la IA
                        if description.strip() == "No description provided.":
                            deepseek = DeepSeekDocumenter()
                            ai_prompt = (f"Genera una breve descripción para el método '{method['name']}' "
                                         f"que recibe los parámetros: {', '.join(args)}.")
                            ai_description = deepseek.generate_documentation(ai_prompt, "")
                            if ai_description and ai_description != "":
                                description = ai_description
                                method['description'] = description  # Actualiza para futuras consultas
                        break
        elif item_type == "Función":
            for func in self.method_descriptions['functions']:
                if func['name'] == item_name:
                    description = func['description']
                    args = func.get('args', [])
                    if description.strip() == "No description provided.":
                        deepseek = DeepSeekDocumenter()
                        ai_prompt = (f"Genera una breve descripción para la función '{func['name']}' "
                                     f"que recibe los parámetros: {', '.join(args)}.")
                        ai_description = deepseek.generate_documentation(ai_prompt, "")
                        if ai_description and ai_description != "":
                            description = ai_description
                            func['description'] = description
                    break
        
        try:
            self.details_text.configure(state='normal')
            self.details_text.delete("1.0", tk.END)
            self.details_text.insert(tk.END, description)
            self.details_text.configure(state='disabled')
        except Exception:
            pass  # En caso la ventana se haya destruido, ignoramos el error

class CodeStructureAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.classes = []
        self.functions = []
        self.method_descriptions = {'classes': [], 'functions': []}

    def visit_ClassDef(self, node):
        class_info = {
            'name': node.name,
            'methods': []
        }
        for body_item in node.body:
            if isinstance(body_item, ast.FunctionDef):
                method_info = {
                    'name': body_item.name,
                    'args': [arg.arg for arg in body_item.args.args],
                    'description': ast.get_docstring(body_item) or "No description provided."
                }
                class_info['methods'].append(method_info)
        self.classes.append(class_info)
        self.method_descriptions['classes'].append(class_info)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        func_info = {
            'name': node.name,
            'args': [arg.arg for arg in node.args.args],
            'description': ast.get_docstring(node) or "No description provided."
        }
        self.functions.append(func_info)
        self.method_descriptions['functions'].append(func_info)
        self.generic_visit(node)

    def get_report(self):
        report = ""
        if self.classes:
            report += "Clases\n" + "="*50 + "\n"
            for cls in self.classes:
                report += f"Clase: {cls['name']}\n"
                if cls['methods']:
                    report += "  Métodos:\n"
                    for method in cls['methods']:
                        args = ", ".join(method['args'])
                        report += f"    - {method['name']}({args})\n"
                report += "\n"
        if self.functions:
            report += "Funciones\n" + "="*50 + "\n"
            for func in self.functions:
                args = ", ".join(func['args'])
                report += f"  - {func['name']}({args})\n"
        if not self.classes and not self.functions:
            report = "No se encontraron clases o funciones en el código proporcionado."
        return report

class KotlinCodeStructureAnalyzer:
    def __init__(self):
        self.classes = []
        self.functions = []
        self.method_descriptions = {'classes': [], 'functions': []}

    def parse(self, code):
        class_pattern = re.compile(r'class\s+(\w+)\s*{')
        method_pattern = re.compile(r'fun\s+(\w+)\s*\(([^)]*)\)')
        description_pattern = re.compile(r'/\*\*(.*?)\*/', re.DOTALL)

        current_class = None
        lines = code.split('\n')
        for i, line in enumerate(lines):
            class_match = class_pattern.search(line)
            if class_match:
                current_class = class_match.group(1)
                class_info = {'name': current_class, 'methods': []}
                self.classes.append(class_info)
                continue

            method_match = method_pattern.search(line)
            if method_match:
                method_name = method_match.group(1)
                args = method_match.group(2).split(',') if method_match.group(2).strip() else []
                args = [arg.split(':')[0].strip() for arg in args]
                description = "No description provided."
                # Buscar comentarios de documentación anteriores
                if i > 0 and description_pattern.search(lines[i-1]):
                    description = description_pattern.search(lines[i-1]).group(1).strip()
                method_info = {
                    'name': method_name,
                    'args': args,
                    'description': description
                }
                if current_class:
                    self.classes[-1]['methods'].append(method_info)
                    self.method_descriptions['classes'].append(self.classes[-1])
                else:
                    self.functions.append(method_info)
                    self.method_descriptions['functions'].append(method_info)

    def get_report(self):
        report = ""
        if self.classes:
            report += "Clases\n" + "="*50 + "\n"
            for cls in self.classes:
                report += f"Clase: {cls['name']}\n"
                if cls['methods']:
                    report += "  Métodos:\n"
                    for method in cls['methods']:
                        args = ", ".join(method['args'])
                        report += f"    - {method['name']}({args})\n"
                report += "\n"
        if self.functions:
            report += "Funciones\n" + "="*50 + "\n"
            for func in self.functions:
                args = ", ".join(func['args'])
                report += f"  - {func['name']}({args})\n"
        if not self.classes and not self.functions:
            report = "No se encontraron clases o funciones en el código proporcionado."
        return report

def analyze_code_dialog(parent):
    dialog = CodeInputDialog(parent, "Analizar Código", "Pega tu código aquí:")
    code = dialog.result
    if code:
        # Preguntar al usuario el lenguaje del código
        language = custom_askstring("Seleccionar Lenguaje", "Ingresa el lenguaje del código (Python/Kotlin):", parent=parent)
        if language is None:
            return  # El usuario canceló la selección
        if language.lower() not in ["python", "kotlin"]:
            messagebox.showerror("Error", "Lenguaje no soportado. Por favor, elige Python o Kotlin.")
            return
        analyzer = CodeAnalyzer(parent, code, language)

def document_code(parent, documenter):
    dialog = DocumentDialog(parent, "Documentar Código", "Ingresa una breve descripción (opcional):")
    user_input = dialog.result
    if user_input is not None:
        code_dialog = CodeInputDialog(parent, "Código a Documentar", "Pega el código que deseas documentar aquí:")
        code_to_document = code_dialog.result
        if code_to_document:
            documented_code = documenter.generate_documentation(user_input, code_to_document)
            if "Error al generar documentación" not in documented_code: #Comprobar si hubo error antes de mostrar la ventana
                doc_window = DocumentedCodeWindow(parent, documented_code)

def document_readme(parent, documenter):
    dialog = DocumentDialog(parent, "Generar README", "Ingresa una breve descripción (opcional):")
    user_input = dialog.result
    if user_input is not None:
        code_dialog = CodeInputDialog(parent, "Código para README", "Pega el código para generar el README:")
        code_to_document = code_dialog.result
        if code_to_document:
            readme_doc = documenter.generate_readme(user_input, code_to_document)
            if "Error al generar README" not in readme_doc:
                doc_window = DocumentedCodeWindow(parent, readme_doc)

def analyze_code():
    # Solicitar al usuario que ingrese el código a analizar
    code = simpledialog.askstring("Código a Analizar", "Introduce el código a analizar:")
    if not code:
        messagebox.showinfo("Análisis cancelado", "No se proporcionó código para analizar.")
        return
    ds = DeepSeekDocumenter()
    prompt = "Analiza el siguiente código y explica su funcionamiento:"
    documentation = ds.generate_documentation(prompt, code)
    # Mostrar el resultado en una ventana emergente
    result_window = tk.Toplevel(root)
    result_window.title("Análisis de Código")
    st = scrolledtext.ScrolledText(result_window, width=80, height=30)
    st.pack(expand=True, fill='both')
    st.insert(tk.END, documentation)

def open_project_window():
    root = tk.Tk()
    root.title("Project Organizer")
    root.geometry("800x600")  # Ajustado para mejor visibilidad
    root.configure(bg="#1a1a2e")  # Fondo oscuro

    projects = load_projects()

    # Crear un Frame central para centrar los widgets con márgenes reducidos
    main_frame = tk.Frame(root, bg="#1a1a2e")
    main_frame.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)  # Reducido de 20 a 5

    # Aplicar estilo ciberpunk
    style = ttk.Style()
    style.theme_use("default")

    # Configurar el estilo para Treeview
    style.configure("Treeview",
                    background="#16213e",
                    foreground="#00ffea",
                    fieldbackground="#16213e",
                    font=("Courier", 12))
    style.configure("Treeview.Heading",
                    background="#0f3460",
                    foreground="#ffffff",
                    font=("Courier", 12, "bold"))

    # Configurar el estilo para el estado 'active' y 'selected'
    style.map("Treeview",
              background=[('active', '#0f3460'), ('selected', '#e94560')],
              foreground=[('active', '#00ffea'), ('selected', '#ffffff')])

    # Configurar el Treeview
    columns = ("Section", "Description", "State")
    tree = ttk.Treeview(main_frame, columns=columns, show='headings', height=15)
    tree.heading("Section", text="Section")
    tree.heading("Description", text="Description")
    tree.heading("State", text="State")

    # Establecer la alineación centrada
    tree.column("Section", anchor="center", width=150)
    tree.column("Description", anchor="center", width=350)
    tree.column("State", anchor="center", width=150)

    # Aplicar estilo al Treeview
    tree.configure(selectmode='browse')
    tree.pack(pady=5, padx=5, fill=tk.BOTH, expand=True)  # Reducido de pady=10 a 5 y agregado fill y expand

    def add_section():
        section = custom_askstring("Añadir Sección", "Ingresa el nombre de la nueva sección:", parent=root)
        if section:
            description = custom_askstring("Añadir Descripción", "Ingresa la descripción:", parent=root)
            state = custom_askstring("Añadir Estado", "Ingresa el estado (no empezado, en progreso, finalizado):",
                                     parent=root)
            if state not in ["no empezado", "en progreso", "finalizado"]:
                state = "no empezado"  # Valor por defecto si la entrada es inválida
            projects.append({
                "section": section,
                "description": description,
                "state": state
            })
            save_projects(projects)
            refresh_treeview()

    def edit_section():
        selected = tree.selection()
        if not selected:
            messagebox.showwarning("Advertencia", "No has seleccionado ninguna sección para editar.")
            return
        item = selected[0]
        index = tree.index(item)
        project = projects[index]
        description = custom_askstring("Modificar Descripción", "Modifica la descripción:",
                                       parent=root, initialvalue=project['description'])
        if description is not None:
            project['description'] = description
        state = custom_askstring("Modificar Estado", "Modifica el estado (no empezado, en progreso, finalizado):",
                                 parent=root, initialvalue=project['state'])
        if state in ["no empezado", "en progreso", "finalizado"]:
            project['state'] = state
        save_projects(projects)
        refresh_treeview()

    def delete_section():
        selected = tree.selection()
        if not selected:
            messagebox.showwarning("Advertencia", "No has seleccionado ninguna sección para borrar.")
            return
        item = selected[0]
        index = tree.index(item)
        confirm = messagebox.askyesno("Confirmar Borrado", "¿Estás seguro de que deseas borrar esta sección?")
        if confirm:
            del projects[index]
            save_projects(projects)
            refresh_treeview()

    def analyze_code():
        # Solicitar al usuario que ingrese el código a analizar
        code = simpledialog.askstring("Código a Analizar", "Introduce el código a analizar:")
        if not code:
            messagebox.showinfo("Análisis cancelado", "No se proporcionó código para analizar.")
            return
        ds = DeepSeekDocumenter()
        prompt = "Analiza el siguiente código y explica su funcionamiento:"
        documentation = ds.generate_documentation(prompt, code)
        # Mostrar el resultado en una ventana emergente
        result_window = tk.Toplevel(root)
        result_window.title("Análisis de Código")
        st = scrolledtext.ScrolledText(result_window, width=80, height=30)
        st.pack(expand=True, fill='both')
        st.insert(tk.END, documentation)

    try:
        documenter = DeepSeekDocumenter()  # Inicializa la clase aquí
    except ValueError as e:
        messagebox.showerror("Error de inicialización", str(e))
        return  # Sale de la función si hay un error en la inicialización

    def document_code_button():
        """
        Solicita al usuario ingresar el código a documentar.
        Si se proporciona el código, se realiza una única llamada a la IA para generar la documentación.
        Si no se ingresa nada, cancela la operación sin abrir múltiples diálogos.
        """
        code = simpledialog.askstring("Código a Documentar", "Introduce el código a documentar:")
        if not code:
            messagebox.showinfo("Documentación cancelada", "No se proporcionó código para documentar.")
            return
        ds = DeepSeekDocumenter()
        prompt = "Documenta el siguiente código, añadiendo comentarios claros a funciones, clases y métodos:"
        documentation = ds.generate_documentation(prompt, code)
        _mostrar_resultado("Documentación del Código", documentation)

    def document_readme_button():
        try:
            with open("main.py", "r") as f:
                code = f.read()
        except Exception as e:
            messagebox.showerror("Error al leer archivo", str(e))
            return
        existing_readme = ""
        if os.path.exists("README.md"):
            with open("README.md", "r") as rf:
                existing_readme = rf.read()
        ds = DeepSeekDocumenter()
        prompt = ("Genera una documentación en formato README para este proyecto. "
                  "Utiliza el siguiente código y la información existente como referencia:\n\n" 
                  f"{existing_readme}\n\nCódigo:\n")
        documentation = ds.generate_documentation(prompt, code)
        # Mostrar el README generado en una ventana emergente para revisión
        result_window = tk.Toplevel(root)
        result_window.title("Generar README")
        st = scrolledtext.ScrolledText(result_window, width=80, height=30)
        st.pack(expand=True, fill='both')
        st.insert(tk.END, documentation)

    def generate_documentation_placeholder(user_input, code):
        # Placeholder para la generación de documentación
        # Implementa aquí la integración con una API de IA como OpenAI
        return """
# Documentación Generada Automáticamente

# Descripción: {}

class Ejemplo:
    def metodo_ejemplo(self, parametro):
        \"\"\"
        Este método realiza una operación importante.
        
        Parámetros:
            parametro (tipo): Descripción del parámetro.
        
        Retorna:
            tipo: Descripción del valor retornado.
        \"\"\"
        pass
""".format(user_input if user_input else "Descripción automática generada.")

    def refresh_treeview():
        for item in tree.get_children():
            tree.delete(item)
        for project in projects:
            tree.insert("", "end", values=(project['section'], project['description'], project['state']))

    # Frame para los botones con márgenes reducidos
    button_frame = tk.Frame(main_frame, bg="#1a1a2e")
    button_frame.pack(pady=5, padx=5, fill=tk.X)  # Cambiado anchor='w' por fill=tk.X

    # Configurar las columnas para que se expandan equitativamente
    num_buttons = 4  # Incrementado a 6 para incluir 'Pasar a README'
    for i in range(num_buttons):
        button_frame.grid_columnconfigure(i, weight=1, uniform="button")

    add_button = tk.Button(button_frame, text="Agregar Sección", command=add_section,
                           bg="#0f3460", fg="#00ffea", font=("Courier", 10, "bold"))
    add_button.grid(row=0, column=0, padx=2, pady=2, sticky='ew')  # Eliminado width=15

    edit_button = tk.Button(button_frame, text="Editar Sección", command=edit_section,
                            bg="#0f3460", fg="#00ffea", font=("Courier", 10, "bold"))
    edit_button.grid(row=0, column=1, padx=2, pady=2, sticky='ew')  # Eliminado width=15

    delete_button = tk.Button(button_frame, text="Borrar Sección", command=delete_section,
                              bg="#e94560", fg="#ffffff", font=("Courier", 10, "bold"))
    delete_button.grid(row=0, column=2, padx=2, pady=2, sticky='ew')  # Eliminado width=15


    document_button = tk.Button(button_frame, text="Documentar", command=document_code_button,
                                bg="#0f3460", fg="#00ffea", font=("Courier", 10, "bold"))
    document_button.grid(row=0, column=3, padx=2, pady=2, sticky='ew')  # Nuevo botón



    refresh_treeview()

    root.mainloop()

if __name__ == "__main__":
    listener_thread = threading.Thread(target=start_listener, daemon=True)
    listener_thread.start()
    open_project_window()
