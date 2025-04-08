import gradio as gr

def procesar_texto(texto: str, request: gr.Request):

    ip_cliente = request.client.host if request else "No disponible"
    user_agent = request.headers.get("user-agent", "Desconocido")

    return f"Texto recibido desde {ip_cliente}\nUser Agent: {user_agent}\nContenido: {texto}"

app = gr.Interface(
    fn=procesar_texto,
    inputs=gr.Textbox(),
    outputs=gr.Textbox(),
    title="Monitor de IPs"
)

app.launch()