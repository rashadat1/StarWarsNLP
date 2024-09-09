import gradio as gr
from theme_classification import ZeroShotClassifier
from character_network import NamedEntityRecognition

def get_themes(theme_list_str, subtitles_path, save_path):
    theme_list = theme_list_str.split(',')
    classifier = ZeroShotClassifier(theme_list)
    output = classifier.get_themes(subtitles_path, save_path)
    
    output = output.T.reset_index()
    output.columns = ['Theme', 'Score']
    return output

def ner(subtitles_path, save_path):
    nerRecog = NamedEntityRecognition()
    ner_output = nerRecog.generate_nerOutput(subtitles_path)
    
    html = nerRecog.draw_Network_Graph(ner_output)
    return html

    
def main():
    with gr.Blocks() as iface:
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Theme Classification with a Zero-Shot Classifier</h1>")
                with gr.Row():
                    with gr.Column():
                        output_chart = gr.BarPlot(
                                    x="Theme",
                                    y="Score",
                                    title="Score for Each Theme",
                                    tooltip=["Theme","Score"],
                                    vertical=True,
                                    width=500,
                                    height=260
                                    )
                    with gr.Column():
                        theme_list = gr.Textbox(label="List of Themes")
                        subtitles_path = gr.Textbox(label="Subtitle Path")
                        save_path = gr.Textbox(label="Save Path")
                        inference_button = gr.Button("Classify")
                        inference_button.click(get_themes, inputs=[theme_list, subtitles_path, save_path], outputs = output_chart)
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Character Network with Named Entity Recognition</h1>")
                with gr.Row():
                    with gr.Column():
                        networkHTML = gr.HTML()
                    with gr.Column():
                        subtitles_path = gr.Textbox(label="Subtitles or Script Path")
                        ner_path = gr.Textbox(label="NER save path")
                        character_network_button = gr.Button("Generate Character Network")
                        character_network_button.click(ner, inputs=[subtitles_path,ner_path],outputs=[networkHTML])

    iface.launch(share=True)
        
if __name__ == "__main__":
    main()