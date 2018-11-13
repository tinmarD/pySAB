import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import tkinter as tk
import tkinter.ttk as ttk
import _pickle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from sklearn.cluster import KMeans
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sab_clustering import *

clustering_algorithms = ['K-Means', 'Affinity Propagation', 'Spectral Clustering', 'DBSCAN']


class TimeFeatureWindow:
    """ Tkinter Interface to visualise TimeFeatures instances.
    Allow to visualise :
     * ERPs
     * ERP images
     * Feature distribution
     * Correlation between hits features and reaction times
     * Clustering
    It takes as input the TimeFeatures instance.
    """
    def __init__(self, time_features, win_size="1200x780"):
        # Time feature attributes
        self.time_features = time_features
        self.feature_sel_pos = np.array([0])
        self.label_keys_sel = list(time_features.label_dict.keys())
        # GUI
        self.root = tk.Tk()
        self.root.geometry(win_size)
        self.container = tk.Frame(self.root)
        self.container.pack(side="top", fill="both", expand=True)
        # List box containing the names of the features
        self.feature_listbox = tk.Listbox(self.container, font="Helvetica 9 bold", selectmode='single')
        self.set_feature_name_list()
        self.feature_listbox.grid(row=2, column=1, sticky='NSEW')
        self.feature_label = tk.Label(self.container, text='Feature Name')
        self.feature_label.grid(row=1, column=1, sticky='S')
        # For each condition in time_features, create a radio button
        self.cond_on = {}
        self.conditions_box = tk.Frame(self.container)
        self.conditions_box.grid(row=2, column=3, sticky='E')
        for cond in list(time_features.label_dict.values()):
            self.cond_on[cond] = tk.IntVar()
            self.cond_on[cond].set(1)
            cb_i = tk.Checkbutton(self.conditions_box, text=cond, variable=self.cond_on[cond],
                                  command=self.onconditionselect)
            cb_i.pack()
        # Clustering options
        self.cluster_frame = tk.LabelFrame(self.container)
        tk.Label(self.cluster_frame, text='Clustering Algo. :').pack()
        self.cluster_algo = clustering_algorithms[0]
        self.cb_cluster_algo = ttk.Combobox(self.cluster_frame, values=clustering_algorithms)
        self.cb_cluster_algo.set(clustering_algorithms[0])
        self.cb_cluster_algo.bind('<<ComboboxSelected>>', self.onclusteralgochanged)
        self.cb_cluster_algo.pack()
        tk.Label(self.cluster_frame, text='N clusters :').pack()
        vcmd = (self.cluster_frame.register(self.onvalidate_nclusters), '%P')
        self.n_clusters = 2
        tk.Entry(self.cluster_frame, text='2', validate='key', validatecommand=vcmd).pack()
        self.cluster_frame.grid(row=3, column=3, sticky='E')
        # Main central graph to display matplotlib figure
        self.f = Figure()
        self.ax, self.cax = self.f.add_subplot(111), []
        self.create_colorbar_axis()
        self.canvas = FigureCanvasTkAgg(self.f, self.container)
        self.set_main_graph()
        # Slider for selecting time point
        self.time_sel_sample = 0
        self.time_slider = tk.Scale(self.container, from_=time_features.tmin, to=time_features.tmax, state='disabled',
                                    resolution=(time_features.tmax-time_features.tmin)/time_features.n_pnts,
                                    command=self.update_time_sel, orient=tk.HORIZONTAL)
        self.time_slider.grid(row=4, column=2, sticky='EW', padx=100, pady=5)
        # Different plotting functions availables
        self.plot_methods = ['erp', 'erp-traces', 'erp-image', 'distribution', 'corr-RT', 'clustering']
        self.plot_method_sel = tk.StringVar()
        self.plot_method_sel.set('erp')
        plot_methods_box = tk.Frame(self.container)
        plot_methods_box.grid(row=3, column=2)
        for plot_method_i in self.plot_methods:
            b = tk.Radiobutton(plot_methods_box, variable=self.plot_method_sel, text=plot_method_i,
                               value=plot_method_i, command=self.onplotmethodselect)
            b.pack(side='left')
        # Callbacks
        self.feature_listbox.bind('<<ListboxSelect>>', self.onfeatureselect)
        # Start the main loop
        self.start()

    def create_colorbar_axis(self):
        """ Create the colorbar axis """
        if not self.cax:
            divider = make_axes_locatable(self.ax)
            self.cax = divider.append_axes('right', size='5%', pad=0.05)

    def destroy_colorbar_axis(self):
        """ Destroy the colorbar axis"""
        if self.cax:
            self.cax.remove()
            self.cax = []

    def update_time_sel(self, val):
        """ Called when the time slider is modified"""
        self.time_sel_sample = self.time_features.time2sample(float(val))
        if self.plot_method_sel.get() in ['distribution', 'clustering', 'corr-RT']:
            self.plot_figure()

    def set_feature_name_list(self):
        """ Set the feature name in the left side listbox"""
        for feature_name_i in self.time_features.feature_names:
            self.feature_listbox.insert('end', feature_name_i)

    def set_main_graph(self):
        """ Set the main center graph """
        self.canvas.show()
        self.canvas.get_tk_widget().grid(row=2, column=2, sticky='NSEW')
        tk.Grid.columnconfigure(self.container, 2, weight=1)
        tk.Grid.rowconfigure(self.container, 2, weight=1)
        frame_toolbar = tk.Frame(self.container)
        frame_toolbar.grid(row=1, column=2)
        NavigationToolbar2TkAgg(self.canvas, frame_toolbar)

    def onconditionselect(self):
        """ Called when the user change the selected conditions """
        label_values = self.time_features.label_dict.values()
        key_sel = []
        for cond_i in list(label_values):
            if self.cond_on[cond_i].get() == 1:
                key_sel.append(self.time_features.get_label_key_from_value(cond_i))
        self.label_keys_sel = key_sel
        self.plot_figure()

    def onfeatureselect(self, evt):
        """ Called when the user select features"""
        # Note here that Tkinter passes an event object to onselect()
        w = evt.widget
        index = w.curselection()
        # feature_name = w.get(index)
        # self.feature_sel_pos = np.where(self.time_features.feature_names == feature_name)[0]
        self.feature_sel_pos = np.array(index, dtype=int)
        self.plot_figure()

    def onplotmethodselect(self):
        """ Called when the user select a different analysis to plot"""
        if self.plot_method_sel.get() in ['clustering']:
            self.feature_listbox['selectmode'] = tk.MULTIPLE
        else:
            self.feature_listbox['selectmode'] = tk.SINGLE
        self.plot_figure()

    def onclusteralgochanged(self, evt):
        """ Called when the user select the clustering algorithm """
        self.cluster_algo = self.cb_cluster_algo.get()

    def onvalidate_nclusters(self, val):
        """ Called when the user changes the number of clusters

        Parameters
        ----------
        val : int
            Number of clusters

        """
        if str.isnumeric(val):
            self.n_clusters = int(val)
            return True
        elif not val:
            self.n_clusters = []
            return True
        else:
            return False

    def plot_figure(self):
        """ Plot the figure given the current selected parameters """
        self.ax.clear()
        if self.feature_sel_pos.size > 0:
            if self.plot_method_sel.get() == 'erp':
                self.destroy_colorbar_axis()
                self.time_features.plot_feature_erp(feature_pos=self.feature_sel_pos, label_keys=self.label_keys_sel,
                                                    ax=self.ax)
                self.time_slider['state'] = 'disabled'
            elif self.plot_method_sel.get() == 'erp-traces':
                self.destroy_colorbar_axis()
                self.time_features.plot_feature_erp(feature_pos=self.feature_sel_pos, label_keys=self.label_keys_sel,
                                                    plot_traces=1, ax=self.ax)
                self.time_slider['state'] = 'disabled'
            elif self.plot_method_sel.get() == 'erp-image':
                self.create_colorbar_axis()
                self.time_features.plot_feature_erpimage(feature_pos=self.feature_sel_pos, label_keys=self.label_keys_sel,
                                                         ax=self.ax, cax=self.cax)
                self.time_slider['state'] = 'disabled'
            elif self.plot_method_sel.get() == 'distribution':
                self.destroy_colorbar_axis()
                self.time_features.plot_feature_distribution(feature_pos=self.feature_sel_pos, label_keys=self.label_keys_sel,
                                                             time_points=self.time_sel_sample, ax=self.ax)
                if self.time_slider['state'] == 'disabled':
                    self.time_slider['state'] = 'normal'
            elif self.plot_method_sel.get() == 'corr-RT':
                self.destroy_colorbar_axis()
                self.time_features.plot_feature_hits_reaction_time(feature_pos=self.feature_sel_pos, label_keys=self.label_keys_sel,
                                                                   time_points=self.time_sel_sample, ax_list=self.ax)
                if self.time_slider['state'] == 'disabled':
                    self.time_slider['state'] = 'normal'
            elif self.plot_method_sel.get() == 'clustering':
                if self.n_clusters:
                    self.create_colorbar_axis()
                    cluster_model = get_clustering_algo(self.cluster_algo, self.n_clusters)
                    self.time_features.cluster_data(cluster_model, do_plot=True, ax=self.ax, feature_pos=self.feature_sel_pos,
                                                    label_keys=self.label_keys_sel, time_points=self.time_sel_sample,
                                                    cb_ax=self.cax)
                    if self.time_slider['state'] == 'disabled':
                        self.time_slider['state'] = 'normal'
            else:
                print('Error - Wrong argument for plot_method_sel : {}'.format(self.plot_method_sel.get()))
        self.canvas.draw()

    def start(self):
        """ Start the application"""
        self.root.mainloop()
        self.plot_figure()

