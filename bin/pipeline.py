from Image import Image
from Metrics import Metrics


def main():
   pass 


if __name__ == "__main__":
    main()
"""
import concurrent.futures
from python_classes import Image, LLM, Processing, Metrics, Alias, Dataset

def process_image(percorso):
    
    Processes a single image and returns the testo and confusion_matrix.

    :param percorso: The file path of the image to process.
    :return: A tuple containing testo and confusion_matrix.
    
    img = Image(percorso)
    gt = img.gt
    testo = LLM.get_testo()
    results = modello_vis(img, testo)

    confusion_matrix = Processing(results, gt).confusion_matrix
    return testo, confusion_matrix

def main(lista_immagini):
    
    Main function to process a list of images using multiprocessing.

    :param lista_immagini: List of image paths to process.
    
    dataset = Dataset()

    # Use ProcessPoolExecutor for parallel processing
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit tasks and get futures
        futures = {executor.submit(process_image, percorso): percorso for percorso in lista_immagini}
        for future in concurrent.futures.as_completed(futures):
            percorso = futures[future]
            try:
                testo, confusion_matrix = future.result()
                
                # Continue with testo and confusion_matrix
                metrics = Metrics(confusion_matrix)
                average_precision = metrics.av_p
                alias = Alias(testo, average_precision)
                dictionary = alias.dictionary
                dataset.save_data(dictionary)
            except Exception as e:
                print(f"Error processing {percorso}: {e}")

    # Load the combined data if needed
    dict = dataset.load('path_to_json')
    print(dict)

if __name__ == "__main__":
    lista_immagini = ['/path/to/image1.jpg', '/path/to/image2.jpg']  # Replace with actual image paths
    main(lista_immagini)

"""
