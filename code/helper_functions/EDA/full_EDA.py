import os
import glob
        
def plot_epochs_all(epochs, directory= os.path.join(os.getcwd(),'data','plots')):
    from . import plot_epochs_image, plot_epochs_TFR, plot_epochs_topomap
    plot_epochs_image(epochs, directory)
    plot_epochs_TFR(epochs, directory)
    plot_epochs_topomap(epochs, directory)

def perform_EDA(subject, run, paradigm, dataset_no, directory = os.path.join(os.getcwd(),'data','plots')):
    from helper_functions import combine_images, add_border_and_title, load_procesed_data, epochs_to_evoked, plot_evoked

    # Ensure the data loading function is defined and available in your workspace.
    data = load_procesed_data(dataset_no, paradigm, subject, run)
    evoked_dict = epochs_to_evoked(data)
    save_dir = os.path.join(os.getcwd(), 'data', 'plots', 'combined')
    last_dir = os.path.join(save_dir,'final')
    os.makedirs(last_dir, exist_ok=True)
    evoked_plot = plot_evoked(evoked_dict, plot=True, save=True, subject=subject, run=run, directory=last_dir)
    evoked_topo = plot_evoked(evoked_dict, plot_topo=True, save=True, subject=subject, run=run, directory=last_dir)
    
    epochs_raw, epochs_raw_cleaned, epochs_raw_autoreject, epochs_raw_cleaned_autoreject, raw_cleaned, raw = data
    epoch_list = [epochs_raw, epochs_raw_cleaned, epochs_raw_autoreject, epochs_raw_cleaned_autoreject]
    for epochs, epochs_name in zip(epoch_list, ['epochs_raw', 'epochs_raw_cleaned', 'epochs_raw_autoreject', 'epochs_raw_cleaned_autoreject']):
        # Example usage
        plot_epochs_all(epochs, directory)        
 
        tfr_png_paths = glob.glob(os.path.join(os.getcwd(), 'data','plots',f's{subject}.{run}_{epochs_name}-**TFR.png'))
        image_png_paths = glob.glob(os.path.join(os.getcwd(), 'data','plots',f's{subject}.{run}_{epochs_name}-*image*.png'))
        topomap_png_paths = glob.glob(os.path.join(os.getcwd(), 'data','plots',f's{subject}.{run}_{epochs_name}-*epochs**topomap*.png'))
        base_dir = os.path.join(os.getcwd(), 'data','plots','combined')
        os.makedirs(base_dir, exist_ok=True)
        tfr_plot_name = os.path.join(os.getcwd(), os.path.join(base_dir, f's{subject}.{run}_{epochs_name}_vcomb_tfr.png'))
        image_plot_name = os.path.join(os.getcwd(), os.path.join(base_dir, f's{subject}.{run}_{epochs_name}_vcomb_image.png'))
        topomap_plot_name = os.path.join(os.getcwd(), os.path.join(base_dir, f's{subject}.{run}_{epochs_name}_vcomb_topomap.png'))

        combine_images(tfr_png_paths, tfr_plot_name, layout='vertical')
        combine_images(image_png_paths, image_plot_name, layout='vertical')
        combine_images(topomap_png_paths, topomap_plot_name, layout='vertical')
        for file in [tfr_plot_name,image_plot_name,topomap_plot_name]:
            add_border_and_title(
                image_path=file,          # Path to the input image
                output_path=file,  # Path to save the output image
                title=f'{epochs_name}',           # Title text
                border_size=0,                    # Border size in pixels
                title_height=50,
                title_size=30,           # Height of the title area in pixels
                border_color='black',              # Border color
                title_color='black',               # Title text color
                title_bg_color='white'              # Title background color
            )
    tfr_png_paths = glob.glob(os.path.join(os.getcwd(), 'data','plots','combined',f's{subject}.{run}_*TFR.png'))
    image_png_paths = glob.glob(os.path.join(os.getcwd(), 'data','plots','combined',f's{subject}.{run}_*image*.png'))
    topomap_png_paths = glob.glob(os.path.join(os.getcwd(), 'data','plots','combined',f's{subject}.{run}_*epochs**topomap*.png'))

    tfr = os.path.join(last_dir,f's{subject}.{run}_tfr_stacked.png')
    image = os.path.join(last_dir,f's{subject}.{run}_image_stacked.png')
    topomap = os.path.join(last_dir,f's{subject}.{run}_topomap_stacked.png')

    for file, ending in zip([tfr_png_paths,image_png_paths,topomap_png_paths],[tfr, image, topomap]):
        combine_images(file, ending, layout='horizontal')    
    

    for file in glob.glob(os.path.join(last_dir, f's{subject}.{run}*.png')):
        add_border_and_title(
            image_path=file,          # Path to the input image
            output_path=file,  # Path to save the output image
            title=f's{subject}.{run}',           # Title text
            border_size=10,                    # Border size in pixels
            title_height=50,
            title_size=30,           # Height of the title area in pixels
            border_color='black',              # Border color
            title_color='white',               # Title text color
            title_bg_color='black'              # Title background color
        )