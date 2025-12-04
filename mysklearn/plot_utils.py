import matplotlib.pyplot as plt

def plot_bar(x_labels, counts, title="", xlabel="", ylabel="Count", rotation=0):
    plt.bar(x_labels, counts, edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation)
    plt.show()

def plot_histogram(values, bins=10, title="", xlabel="", ylabel="Count"):
    plt.hist(values, bins=bins, edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot_scatter(x_values, y_values, title="", xlabel="", ylabel=""):
    clean_x = []
    clean_y = []
    for x, y in zip(x_values, y_values):
        try:
            fx = float(x)
            fy = float(y)
            clean_x.append(fx)
            clean_y.append(fy)
        except:
            continue

    if len(clean_x) == 0 or len(clean_y) == 0:
        print(f"Warning: No valid data to plot for {title}")
        return

    plt.scatter(clean_x, clean_y, alpha=0.6, edgecolors="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

def plot_boxplot(groups, data, title="", xlabel="", ylabel=""):
    
    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=groups, patch_artist=True,
                boxprops=dict(facecolor="lightblue", color="black"),
                medianprops=dict(color="red", linewidth=1.5),
                whiskerprops=dict(color="black"),
                capprops=dict(color="black"))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()