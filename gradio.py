import gradio as gr

def main():
    with gr.Blocks() as iface:
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Theme Classification with a Zero-Shot Classifier</h1>")
            with gr.Row():
                with gr.Column():
                    plot = gr.BarPlot()
                with gr.Column():
                    theme_list = gr.Textbox(label="List of Themes")
                    subtitles_path = gr.Textbox(label="Subtitle Path")
                    save_path = gr.Textbox(label="Save Path")
                    inference_button = gr.Button("Classify")
    iface.launch(share=True)
        
if __name__ == "__main__":
    main()