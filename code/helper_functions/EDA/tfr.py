def epochs_tfr(tfr=None, subject=None,run=None, show=True, save=False, 
                 extra_mark=None, grand=False, grand_tfr_left=None, grand_tfr_right=None,
                 freq_bands_of_interest=None):
    # Convert TFR to DataFrame in long format
    
    if grand:
        df = grand_tfr_right.to_data_frame(time_format=None, long_format=True)
        df['condition'] = 'right_hand'

        df_2 = grand_tfr_left.to_data_frame(time_format=None, long_format=True)
        df_2['condition'] = 'left_hand'
        df = pd.concat([df, df_2], ignore_index=True)
    else:
        df = tfr.to_data_frame(time_format=None, long_format=True)

    freq_bounds = {"_": 0, "delta": 3, "theta": 7, "alpha": 13, "beta": 35, "gamma": 140}
    
    # Map frequencies to bands
    df["band"] = pd.cut(
        df["freq"], list(freq_bounds.values()), labels=list(freq_bounds)[1:]
    )

    # Filter relevant frequency bands
    if freq_bands_of_interest is None:
       freq_bands_of_interest = ["delta", "theta", "alpha", "beta"]
    df = df[df.band.isin(freq_bands_of_interest)]
    df["band"] = df["band"].cat.remove_unused_categories()

    # Order channels
    df["channel"] = df["channel"].cat.reorder_categories(["C3", "Cz", "C4"], ordered=True)

    # Create the FacetGrid plot
    g = sns.FacetGrid(df, row="band", col="channel", margin_titles=True)
    g.map(sns.lineplot, "time", "value", "condition", n_boot=10)

    # Add vertical and horizontal reference lines
    axline_kw = dict(color="black", linestyle="dashed", linewidth=0.5, alpha=0.5)
    g.map(plt.axhline, y=0, **axline_kw)
    g.map(plt.axvline, x=0, **axline_kw)

    # Set plot limits and labels
    g.set(ylim=(None, 1.5))
    g.set_axis_labels("Time (s)", "ERDS")
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    
    # Add a legend
    g.add_legend(ncol=2, loc="lower center")
    
    # Adjust the subplot layout
    g.fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.08)
    
    # Save the plot if requested
    if save:
        if extra_mark == None:
            extra_mark = ''
        elif extra_mark != None:
            extra_mark = extra_mark + "_"
            
        path = os.path.join(os.getcwd(), 'data', 'plots', 'erds', f'{subject:02}')
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, f's{subject:02}.{run:02}_{extra_mark}ERDS_lineplot.png')
        g.savefig(save_path)
        add_border_and_title(save_path, save_path, f's{subject:02}.{run:02}_{extra_mark}',
                             border_size=0,title_color='black',title_bg_color='white')
        
    # Show the plot if requested
    if show:
        plt.show()

    # Close the plot to avoid memory issues with multiple plots
    plt.close()
