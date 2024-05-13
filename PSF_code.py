import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.modeling import models, fitting

x_dimension = 2721
y_dimension = 1956

stars_file = get_pkg_data_filename("stars.fits")
fits.info(stars_file)
stars_data = fits.getdata(stars_file, ext = 0)
original_image = stars_data
plt.imshow(stars_data, cmap = "gray")
plt.show()

def box(x_centre, y_centre, dimention, stars_data):
    box = np.zeros((dimention, dimention), dtype = float)
    for i in range(dimention):
        for j in range(dimention):
            box[i][j] = stars_data[y_centre - dimention // 2 + i][x_centre - dimention // 2 + j]
    normal_box = box / np.max(box)
    return normal_box
    
def ring(radius, box):
    n = 0
    count = 0
    for y in range(len(box)):
        for x in range(len(box[y])):
            distance = (x - (len(box) // 2)) ** 2 + (y - (len(box) //  2)) ** 2
            d = np.sqrt(distance) - radius
            if abs(d) < 0.5:
               count += box[y][x]
               n += 1
    return count / n

def ring_intensity_plot(box):
    data = []
    for i in range(int(len(box) / 2)):
        data.append(ring(i, box))
    r_line = [j for j in range(len(data))]
    return(data, r_line)

def circle(coordinates, radius, image):
    for y in range(2 * radius + 1):
        for x in range(2 * radius + 1):
            distance = (x - radius) ** 2 + (y - radius) ** 2
            d = np.sqrt(distance) - radius
            if abs(d) < 0.5:
                image[coordinates[1] - radius + y][coordinates[0] - radius + x] = 1
            
    
def distance(xy1, xy2):
    return(np.sqrt((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2))

def star_center_locator(sample_pixel_coordinate, data, dimention):
    x_bar = 0
    y_bar = 0
    tot_intensity = 0
    for i in range(dimention):
        for j in range(dimention):
            intensity = data[i - 5 + sample_pixel_coordinate[1]][j - dimention // 2 + sample_pixel_coordinate[0]]
            tot_intensity += intensity
    for i in range(dimention):
        for j in range(dimention):
            intensity = data[i - 5 + sample_pixel_coordinate[1]][j - dimention // 2 + sample_pixel_coordinate[0]]
            x_bar += (j - dimention // 2 + sample_pixel_coordinate[0]) * intensity / tot_intensity
            y_bar += (i - 5 + sample_pixel_coordinate[1]) * intensity / tot_intensity
    x_bar = int(round(x_bar))
    y_bar = int(round(y_bar))
    return([x_bar, y_bar])

def star_recognition(data):
    changable_data = data
    image_data = data
    coordinates = []
    median = np.median(data)
    std = np.std(data)
    for y in range(len(data)):
        if y % 100 == 0:
            print("Processing : ", round(((y / (y_dimension)) * 100), 2), "%")
        for x in range(len(data[y])):
            if data[y][x] >= median + 6 * std:
                coordinates.append([x, y])
                image_data[y][x] = 1
            else:
                changable_data[y][x] = 0
                image_data[y][x] = 0
    print("Processing : ", 100, "%")
    corrected_coordinates = []
    for k in coordinates:
        count = 0
        if k[0] >= 50 and k[0] <= x_dimension - 50 and k[1] >= 50 and k[1] <= y_dimension - 50:
            for i in range(3):
                for j in range(3):
                    if changable_data[k[1] - 1 + i][k[0] - 1 + j] > 0:
                        count += 1
        if count == 6:
            corrected_coordinates.append(k)
    star_loc = [corrected_coordinates[0]]
    print("Deleted single pixels.")
    for k in corrected_coordinates:
        D = []
        for l in star_loc:
            D.append(distance(k, l))
        if np.min(D) >= 20:
            star_loc.append(k)
    center_loc = []
    print("Processing image.")
    for i in star_loc:
        if i[0] >= 50 and i[0] <= x_dimension - 50 and i[1] >= 50 and i[1] <= y_dimension - 50:
            center_loc.append(star_center_locator(i, data, 30))
            circle(center_loc[-1], 40, image_data)
    print("Calculation Complete.")
    return(center_loc, image)



star_loc, image = star_recognition(stars_data)
plt.imshow(image, cmap = "gray")
plt.title("Star Recognition")
plt.show()
n = 0
sample = [5, 6, 10, 14, 16, 17, 18, 19, 24, 28, 29, 30, 31, 32, 34, 37, 38, 39, 40, 41, 42, 45, 54, 62, 63, 70, 73, 79, 84]
sigma = []
for j in sample:
    data, r_line = ring_intensity_plot(box(star_loc[j][0], star_loc[j][1], 40, stars_data))
    plt.plot(r_line, data, ".", label = "Intensity for Star Number " + str(j + 1))
    line_init = models.Gaussian1D()
    fit = fitting.LevMarLSQFitter()
    x = [0.01 * i for i in range(2000)]
    d_line = []
    for k in r_line:
        d_line.append(-k)
    d_line.reverse()
    d_line = d_line + r_line
    d_data = []
    for l in data:
        d_data.append(l)
    d_data.reverse()
    d_data = d_data + data
    fitted_line = fit(line_init, d_line, d_data)
    plt.plot(x, fitted_line(x), label = "Fitted Model Star Number " + str(j + 1))
    sigma.append(fitted_line.stddev.value)
    print("Star Number ", j + 1, "Std = ", sigma[-1])
    plt.legend()
    n += 1
    if n % 5 == 0:
        plt.xlabel("Radius")
        plt.ylabel("Intensity")
        plt.title("Stellar Intensity for 5 stars")
        plt.show()
plt.show()

print("Median = ", np.median(sigma))
counts, bins = np.histogram(sigma, bins = 15)
plt.stairs(counts, bins)
plt.title("Histogram of Standard Deviation for PSF")
plt.xlabel("Standard Devation")
plt.ylabel("Frequency")
plt.show()





