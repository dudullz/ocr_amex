import PyPDF2
import os, sys
import cv2
import shutil
import numpy as np
import math
from expense_categories import BudgetCategories as bc
from pprint import pprint

categories = {}     ## in format of {sub category : main category}
expense_tot = {}    ## total spending under each main category
for k,v in bc.items():
    expense_tot[k] = 0.0
    for val in v:
        categories[val] = k
pprint(categories)

######### 1. first test PyPDF2 library to extract information from pdf ################
def extract_text_from_pdf(pdf_file_path):
    # Open the PDF file
    with open(pdf_file_path, 'rb') as file:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(file)

        # Initialize an empty string to store text
        text = ''

        i = 1
        # Loop through each page in the PDF
        for page in pdf_reader.pages:
            print('\t{:=^20}'.format("Page " +  str(i)))
            # Extract text from the page
            text1 = page.extract_text()
            text += text1 + "\n"
            
            i += 1

        return text

# Example usage
pdf_path = 'c:/Personal/116/Amex/2023-12-19.pdf'  # Replace with your PDF file path
pdf_path = 'C:/Programs/Tasks/PDFs/page_3.pdf'
# extracted_text = extract_text_from_pdf(pdf_path)
# print(extracted_text)

######### 2. save each page into a separate pdf file ################
## The saved PDFs are not used for further processing below.
def split_pdf_into_pages(pdf_path, output_folder):
    # Open the PDF file in read-binary mode
    with open(pdf_path, 'rb') as file:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(file)

        # Get the number of pages using len()
        num_pages = len(pdf_reader.pages)

        # Iterate through each page in the PDF
        for page_num in range(num_pages):
            # Create a PDF writer object
            pdf_writer = PyPDF2.PdfWriter()

            # Get a specific page
            page = pdf_reader.pages[page_num]

            # Add page to the PDF writer
            pdf_writer.add_page(page)

            # Output file name for each page
            output_filename = f"{output_folder}/page_{page_num + 1}.pdf"

            # Save the page as a new PDF file
            with open(output_filename, 'wb') as output_file:
                pdf_writer.write(output_file)

            print(f"Saved: {output_filename}")

pdf_path = 'c:/Personal/116/Amex/2023-12-19.pdf'
pdf_path = 'c:/Personal/116/Amex/2024-04-19.pdf'  # Replace with your PDF file path
# pdf_path = 'C:/Programs/Tasks/PDFs/page_3.pdf'
output_folder = '0. output_pdf'  # Replace with your desired output folder path
# Construct the correct result path
# os.makedirs(output_folder, exist_ok=True)
# split_pdf_into_pages(pdf_path, output_folder)


######### 3. convert each page to png image then save ################
import os
from PyPDF2 import PdfReader
from pdf2image import convert_from_path

def convert_pdf_to_png(pdf_file_path, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Read the PDF file
    pdf_reader = PdfReader(pdf_file_path)
    num_pages = len(pdf_reader.pages)

    # Convert each page to an image
    for page_number in range(num_pages):
        # Convert the current page to image
        # images = convert_from_path(pdf_file_path, first_page=page_number + 1, last_page=page_number + 1, poppler_path= r"C:/Programs/Tasks/poppler-23.11.0/Library/bin")
        images = convert_from_path(pdf_file_path, first_page=page_number + 1, last_page=page_number + 1, poppler_path= r"C:/Developer/CVML/poppler-25.11.0/Library/bin")
        # Save the image
        for image in images:
            image_path = os.path.join(output_folder, f"page_{page_number + 1}.png")
            image.save(image_path, 'PNG')

# Example usage
pdf_path = 'c:/Personal/116/Amex/2024-11-19.pdf'  # Replace with your PDF file path
pdf_path = 'c:/Personal/116/Amex/2024-10-19.pdf'  # Replace with your PDF file path
pdf_path = 'c:/Personal/116/Amex/2024-09-19.pdf'  # Replace with your PDF file paths
pdf_path = 'c:/Personal/116/Amex/2024-08-19.pdf'  # Replace with your PDF file path
pdf_path = 'c:/Personal/116/Amex/2024-07-19.pdf'  # Replace with your PDF file path
pdf_path = 'c:/Personal/116/Amex/2024-06-19.pdf'  # Replace with your PDF file path
pdf_path = 'c:/Personal/116/Amex/2024-05-19.pdf'  # Replace with your PDF file path
pdf_path = 'c:/Personal/116/Amex/2024-04-19.pdf'  # Replace with your PDF file path
pdf_path = 'c:/Personal/116/Amex/2024-03-19.pdf'  # Replace with your PDF file path
pdf_path = 'c:/Personal/116/Amex/2024-02-19.pdf'  # Replace with your PDF file path
pdf_path = 'c:/Personal/116/Amex/2024-01-19.pdf'  # Replace with your PDF file path
pdf_path = 'c:/Personal/116/Amex/2025-01-19.pdf'  # Replace with your PDF file path
pdf_path = 'c:/Personal/116/Amex/2025-02-19.pdf'  # Replace with your PDF file path
# pdf_path = 'c:/Personal/116/Amex/2025-03-19.pdf'  # Replace with your PDF file path
# pdf_path = 'c:/Personal/116/Amex/2025-04-19.pdf'  # Replace with your PDF file path
# pdf_path = 'c:/Personal/116/Amex/2025-05-19.pdf'  # Replace with your PDF file path
# pdf_path = 'c:/Personal/116/Amex/2025-06-19.pdf'  # Replace with your PDF file path
# pdf_path = 'c:/Personal/116/Amex/2025-07-19.pdf'  # Replace with your PDF file path
# pdf_path = 'c:/Personal/116/Amex/2023-12-19.pdf'  # Replace with your PDF file path
# pdf_path = 'c:/Users/longzl/Downloads/2023-07-27_Statement.pdf'  # Replace with your PDF file path
# pdf_path = 'C:/Programs/Tasks/PDFs/page_2.pdf'
f = os.path.basename(pdf_path)
title = os.path.splitext(f)
output_dir = '1. output_images'  # Replace with your desired output directory

# Remove the output Directory
try:
    # os.rmdir(output_path)
    shutil.rmtree(output_dir)        
    print("Directory has been removed successfully")
except OSError as error:
    print(error)
    print("Directory can not be removed")
# Ensure the output folder exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
convert_pdf_to_png(pdf_path, output_dir)

######### 4. continue to OCR each image and save results to ################
## NOT the optimised results. Just run tesseract on the whole image and save the outputs.
import json
import os
import xml.etree.ElementTree as ET
from PIL import Image
import pytesseract
from pytesseract import Output

def ocr_image_to_text(image_path):
    # Open the image file
    with Image.open(image_path) as img:
        # Use pytesseract to do OCR on the image
        text = pytesseract.image_to_string(img)
    return text

def save_to_json(text, json_file_path):
    # Save the text to a JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump({"text": text}, json_file)

def save_to_xml(text, xml_file_path):
    # Create an XML element
    root = ET.Element("data")
    ET.SubElement(root, "text").text = text

    # Save the text to an XML file
    tree = ET.ElementTree(root)
    tree.write(xml_file_path)

# Directory containing images
image_dir = '1. output_images'  # Replace with your image directory
output_dir = '2.1 output_ocr'  # Replace with your output directory
# Remove the output Directory
try:
    # os.rmdir(output_path)
    shutil.rmtree(output_dir)        
    print("Directory has been removed successfully")
except OSError as error:
    print(error)
    print("Directory can not be removed")
# Ensure the output folder exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print('\n\t[4. OCR each image save results to "2.1 output_ocr" folder (not used)]')
# Process each image in the directory
for image_file in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_file)
    text = ocr_image_to_text(image_path)

    # Create JSON and XML file paths
    base_filename = os.path.splitext(image_file)[0]
    json_path = os.path.join(output_dir, f'{base_filename}.json')
    xml_path = os.path.join(output_dir, f'{base_filename}.xml')

    # Save results to files
    save_to_json(text, json_path)
    save_to_xml(text, xml_path)


######### 5. OCR each image with layout elements such as paragragh, word, text, BBoxes etc. #############
    # Open the image file
    # with Image.open(image_path) as img:
    #     # Use pytesseract to do OCR on the image
    #     d = pytesseract.image_to_data(img, output_type=Output.DICT)
    #     print(d.keys())

    #     img1 = cv2.imread(image_path)
    #     n_boxes = len(d['text'])
    #     for i in range(n_boxes):
    #         if int(d['conf'][i]) > 60:
    #             (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    #             img = cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 2)

    #     # cv2.imshow('img', img1)
    #     # cv2.waitKey(0)
                
    #     # plot the boxes and save image.
    #     base_filename = os.path.splitext(image_file)[0]
    #     boximage_path = os.path.join(output_dir, f'{base_filename}.bmp')
    #     cv2.imwrite(boximage_path, img1)


######### 6. Apply conventional image processing technique to prepare for line detection ################
# Directory containing images
print('\n\t[6. Apply conventional IP techniques to prepare for line detection]')
image_dir = '1. output_images'  # Replace with your image directory
output_dir = '2.2 output_processed'  # Replace with your output directory
# Remove the output Directory
try:
    # os.rmdir(output_path)
    shutil.rmtree(output_dir)        
    print("Directory has been removed successfully")
except OSError as error:
    print(error)
    print("Directory can not be removed")
# Ensure the output folder exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for image_file in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_file)
    img_orig = cv2.imread(image_path)
    img_grey = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
    img_bina = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
    img_bina = ~img_bina

    ## Create the images that will use to extract the horizonta and vertical lines
    img_hcopy = img_bina.copy()
    img_vcopy = img_bina.copy()

    scale = 15 # play with this variable in order to increase/decrease the amount of lines to be detected

    ########### [horiz] ###########
    # Specify size on horizontal axis
    horizontalsize = img_bina.shape[1] // scale
    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT,(horizontalsize,1))

    # Apply morphology operations
    img_hcopy = cv2.erode(img_hcopy, horizontalStructure)
    img_hcopy = cv2.dilate(img_hcopy, horizontalStructure)
    # dilate(horizontal, horizontal, horizontalStructure, Point(-1, -1)); // expand horizontal lines

    ########### [vert] ###########
    # Specify size on vertical axis
    rows = img_vcopy.shape[0]
    verticalsize = rows // 60
    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    # Apply morphology operations
    img_vcopy = cv2.erode(img_vcopy, verticalStructure)
    img_vcopy = cv2.dilate(img_vcopy, verticalStructure)

    base_filename = os.path.splitext(image_file)[0]
    greyimage_path = os.path.join(output_dir, f'{base_filename}-grey.bmp')
    binaimage_path = os.path.join(output_dir, f'{base_filename}-bin.bmp')
    hlineimage_path = os.path.join(output_dir, f'{base_filename}-hlines.bmp')
    vlineimage_path = os.path.join(output_dir, f'{base_filename}-vlines.bmp')

    # cv2.imwrite(greyimage_path, img_grey)
    # cv2.imwrite(binaimage_path, img_bina)
    cv2.imwrite(hlineimage_path, img_hcopy)
    # cv2.imwrite(vlineimage_path, img_vcopy)

    print('image size {}x{}, se_h: {}, se_v:{}'.format(img_vcopy.shape[1], img_vcopy.shape[0], horizontalsize, verticalsize))


######### 7. detect (by hough transform) and refine horizontal lines (merge neibouring lines) ################
# Directory containing images
print('\n\t[7. OCR between horizontal lines]')
image_dir = '2.2 output_processed'  # Replace with your image directory
output_dir = '3. output_hlines'  # Replace with your output directory
# Remove the output Directory
try:
    # os.rmdir(output_path)
    shutil.rmtree(output_dir)        
    print("Directory has been removed successfully")
except OSError as error:
    print(error)
    print("Directory can not be removed")
# Ensure the output folder exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

## quick (and dirty) way to validate the detected horizontal lines in the statements
## just two simple conditions:
## 1. line length is greater than 2/3 of image width
## 2. horizontal (y difference < 5 pixels)
from utilities.datatype_line import FlowchartLine
from utilities.is_hline import is_horizontal_line
from utilities.merge_neighbour_lines import LineMerger
# angle threshold for h/v line checking and line merging
delta_rad = 0.1	#  5.72958 degrees
delta_deg = 5	#  5 degrees
# distance threshold for line merging
delta_rho = 10
# distance threshold for line merging between ending points of two line segments
delta_ptdist = 20

def VerifyHLines(x1,y1, x2,y2, width):
    fline = FlowchartLine(x1,y1,x2,y2)
    fline.calcFeatures()

    if not is_horizontal_line(fline, delta_rad):
        return None

    dist = math.sqrt( (x1-x2)**2 + (y1-y2)**2 )
    # length = cv2.norm((x1, y1) - (x2, y2))
    print(f'P1 {x1,y1}, P2{x2,y2}, Line Length', dist)
    if dist < width and dist >= (4*width / 5) and abs(y2-y1) < 5:
        return fline
    else:
        return None

custom_config = r'--oem 3 --psm 6'

## specify column separators based on AMEX and HSBC layout
month_total = 0.0 # total amount from valid (found in pre-defined category) OCR numbers
new_spend = -999.0 # New spend number reported by bank statement (only 1 item, if OCR'd successfully)
unknown_spend = {}    ## regard as invalid spending, e.g. credit, unknown category
for image_file in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_file)
    img_orig = cv2.imread(image_path)
    print(f'\n\t=== Read Image {image_path} ===')
    print(f'Image size: {img_orig.shape}, nChannels: {img_orig.ndim}')

    ## initialise in-page varialbes
    page_total = 0.0   # sum of OCR'd items in this page
    valid_items = 0
    ## initialise dictionary with {category : integer}
    expense_cur = {}    ## current expense in this page
    for k,v in bc.items():
        expense_cur[k] = 0.0

    dst = cv2.Canny(img_orig, 50, 200, None, 3)
    # Copy edges to the images that will display the results in BGR
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    w = img_orig.shape[1]
    # https://answers.opencv.org/question/206392/calculate-slope-length-and-angle-of-a-specific-part-side-line-on-a-contour/
    #########################################################
    ## a. Detect lines with normal Hough Line Detection.  ###
    #########################################################
    print('\t[ Hough Line Detection ]')
    flines_horizont = []
    lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
    
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))

            rlt = VerifyHLines(pt1[0], pt1[1], pt2[0], pt2[1], w)
            print(rlt)
            if rlt is not None:
                flines_horizont.append(rlt)

    h_cnt = 0
    if len(flines_horizont) > 1:
        lmerger = LineMerger()
        flines_horizont_new = lmerger.merge(flines_horizont, delta_rad, delta_rho, delta_ptdist)
        for l in flines_horizont_new:
            if l.rho == 0 and l.theta == -100:
                continue
            cv2.line(cdst, (l.x1,l.y1), (l.x2,l.y2), (0,255,0), 1, cv2.LINE_AA)
            h_cnt += 1

    if lines is not None:
        print('Detected %d Hough Lines' % len(lines))
    else:
        print('Detected 0 Hough Lines')
    print('Valid H Lines: ', len(flines_horizont))
    print(h_cnt, " horizont lines [After Merging]\n" )
    ##########################################################
    ## b. Detect lines with Probabilistic Hough Transform. ###
    ##########################################################
    print('\t[ a. Probabilistic Hough Line Detection ]')
    flines_horizont = []
    linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
    
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]

            rlt = VerifyHLines(l[0], l[1], l[2], l[3], w)
            print(rlt)
            if rlt is not None:
                flines_horizont.append(rlt)

    h_cnt = 0
    ocr_lines = []  # FlowchartLine 
    if len(flines_horizont) > 1:
        lmerger = LineMerger()
        flines_horizont_new = lmerger.merge(flines_horizont, delta_rad, delta_rho, delta_ptdist)
        for l in flines_horizont_new:
            if l.rho == 0 and l.theta == -100:
                continue
            cv2.line(cdstP, (l.x1,l.y1), (l.x2,l.y2), (0,255,0), 5, cv2.LINE_AA)
            ocr_lines.append(l)
            h_cnt += 1
    
    if linesP is not None:
        print('Detected %d Probabilistic Hough Lines' % len(linesP))
    else:
        print('Detected 0 Probabilistic Hough Lines')
    print('Valid H Lines: ', len(flines_horizont))
    print(h_cnt, " horizont lines [After Merging]\n")

    # cv2.imshow("Source", img_orig)
    # cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    # cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

    base_filename = os.path.splitext(image_file)[0]
    orig_img_name = base_filename[:-7] + '.png'   ## get xxx from xxx-hlines.bmp
    houghlineimage_path = os.path.join(output_dir, f'{base_filename}-HoughLines.bmp')
    phoughlineimage_path = os.path.join(output_dir, f'{base_filename}-PHoughLines.bmp')
    cv2.imwrite(houghlineimage_path, cdst)
    cv2.imwrite(phoughlineimage_path, cdstP)

    ## skip this image if didn't find valid H lines
    if h_cnt < 2:   ## need at least two Horizontal lines to proceed
        print('\t\tValid items in this page:', valid_items, '; Total in page: {}, New Spend:{}\n\n'.format(page_total, new_spend))
        continue

    ##########################################################
    ## c. sort out horizontal lines from top to bottom (based on Y values) ###
    ##########################################################
    ## at least need two lines to have valid OCR area inbetween
    print('\t[ b. OCR between %d lines ]'% len(ocr_lines))
    ys = []
    for i, line in enumerate(ocr_lines):
        print(i, line.x1, line.y1)
        ys.append(line.y1)
    
    line_sorted = []    ## reorganise the lines based on sorted y1 values
    sort_index = np.argsort(ys)
    for idx in sort_index:
        line_sorted.append( ocr_lines[idx] )
    print('\tAfter Sorted based on y1')
    for i, line in enumerate(line_sorted):
        print(i, line.x1, line.y1)
        ys.append(line.y1)
    
    ##########################################################
    ## d. divide further each rectangle area between lines ###
    ##########################################################
    ## for Amex layout, divide each region into 4 segments,
    ## corresponding to 'Transcation Date, Process Date, Transaction Details, and Amount £'
    ## the width of the 1st, 2nd and last segment in term of number of pixels
    amex_width = [100, 100, 150]
    x1 = line_sorted[0].x1 + amex_width[0]
    x2 = x1 + amex_width[1]
    x3 = line_sorted[0].x2 - amex_width[2]
    y_top = line_sorted[0].y1
    y_bot = line_sorted[-1].y1

    n = len(line_sorted)
    if n > 1:
        orig_img_dir = '1. output_images'  ## now we need to load original images for OCR
        image_path = os.path.join(orig_img_dir, orig_img_name)
        orig_img = cv2.imread(image_path)
        print(orig_img.shape, orig_img.ndim)
        img_copy = np.copy(orig_img)    ## image copy for drawing

        ## DO NOT USE sum !!!
        ## https://stackoverflow.com/questions/6929777/typeerror-float-object-is-not-callable
        ##  For example, somewhere in your code you define a variable as:
        ##          sum = 0
        ## Maybe to use it as an accumulator variable in global dataframe. 
        ## Now, later when you're defining a function in which you want to call the inbuilt function sum() , 
        ## its gonna give an type error as you have over-written an in-built function name. 
        ## That's why, you should avoid the use in-built function names like str, range, sum, etc.. as one of the variable names in your code.
        
        seg1 = []   # Transaction Date
        seg2 = []   # Process Date
        seg3 = []   # Description
        seg4 = []   # Amount
        for l, line1 in enumerate(line_sorted):
            ## need two lines for ROI region
            if l == (n-1):
                break
            print('\t[Parse Line {}]'.format(l+1))
            line2 = line_sorted[l+1]
            img_roi = orig_img[line1.y1:line2.y1, line1.x1:line2.x2]
            cv2.rectangle(img_copy, (line1.x1, line1.y1), (line2.x2, line2.y2), (0,0,255))
            # print(img_roi.shape, img_roi.ndim)

            ## Getting boxes around each character detected by tesseract during OCR.
            ## https://nanonets.com/blog/ocr-with-tesseract/
            
            # boxes = pytesseract.image_to_boxes(img_roi) 
            # for b in boxes.splitlines():
            #     b = b.split(' ')
            #     img = cv2.rectangle(img_roi, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

            ## Getting boxes around words instead of characters. image_to_data function with output type specified with pytesseract Output
            ## Using this dictionary, we can get each word detected, their bounding box information, the text in them, and the confidence scores for each.
            ## dict_keys(['level', 'page_num', 'block_num', 'par_num', 'line_num', 'word_num', 'left', 'top', 'width', 'height', 'conf', 'text'])
            ## https://nanonets.com/blog/ocr-with-tesseract/
            d = pytesseract.image_to_data(img_roi, output_type=Output.DICT)
            # print(d.keys())
            n_boxes = len(d['text'])
            for i in range(n_boxes):
                if int(d['conf'][i]) > 60:
                    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                    # cv2.rectangle(img_roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

            ##########################################################
            ## e. NOW OCR each segment region between two lines ###
            ##########################################################
            ## calculate the relative coordinates within img_roi
            roi_seg1 = orig_img[line1.y1:line2.y1, line1.x1:x1]
            roi_seg2 = orig_img[line1.y1:line2.y1, x1:x2]
            roi_seg3 = orig_img[line1.y1:line2.y1, x2:x3]
            roi_seg4 = orig_img[line1.y1:line2.y1, x3:(line1.x2+10)]    # add 10 pixel to avoid cut-off useful text
            cv2.imwrite('seg4.bmp', roi_seg4)

            ## a. first OCR the whole line
            line_rlt = pytesseract.image_to_string(img_roi)
            ## b. then OCR each segment within the line
            s1 = pytesseract.image_to_string(roi_seg1)
            s2 = pytesseract.image_to_string(roi_seg2)
            s3 = pytesseract.image_to_string(roi_seg3)
            s4 = pytesseract.image_to_string(roi_seg4)
            s4 = s4.replace(',','')
            ## c. check OCR'd results
            if 'Total new spend transactions for LONGZHEN LI' in line_rlt and s4 != "":
                new_spend = float(s4.replace(',', ''))

            if s1 == "" or s2 == "" or s3 == "" or s4 == "":
                continue
            if 'CR' in s4:  ## credit does not count.
                continue
            seg1.append(s1)
            seg2.append(s2)
            seg3.append(s3)
            seg4.append(s4.replace(" ", ""))    ## remove all white spaces, as some item shows as '10 1.14\n'
            # seg4.append(s4.replace(",", ""))    ## also remove comma, as some item shows as '10 1.14\n'
            print('Column1: ', seg1)
            print('Column4: ', seg4)
            print()

            ## visualise the segments
            h, w, c = img_roi.shape
            cv2.line(img_roi, (amex_width[0],0), (amex_width[0],h), (255,0,0))
            cv2.line(img_roi, (amex_width[0]+amex_width[1],0), (amex_width[0]+amex_width[1],h), (255,0,0))
            cv2.line(img_roi, (w - amex_width[2],0), (w - amex_width[2],h), (255,0,0))
            roi_path = os.path.join(output_dir, f'{base_filename}-{l+1}.bmp')
            cv2.imwrite(roi_path, img_roi)

    ## draw vertical separators
    cv2.line(img_copy, (x1, y_top), (x1, y_bot), (0,255,0), 2 )
    cv2.line(img_copy, (x2, y_top), (x2, y_bot), (0,255,0), 2 )
    cv2.line(img_copy, (x3, y_top), (x3, y_bot), (0,255,0), 2 )
    roiimage_path = os.path.join(output_dir, f'{base_filename}-ROIs.bmp')
    cv2.imwrite(roiimage_path, img_copy)

    print('Total Non-Empty Items (4 columns): ', len(seg1), len(seg2), len(seg3), len(seg4))
    ## No OCR result for first two columns, skip this page.
    if (len(seg1) == len(seg2) == 0):
        continue
    if seg1[0] == 'Transaction\nDate\n' and seg2[0] == 'Process\nDate\n':
        print('\t=== Find Valid Transaction Page ===')
        
        for i, des in enumerate(seg3):
            ## OCR result: BOOTS THE CHEMIST       READING  ---> BOOTS THECHEMIST READING
            print('\t[Check for No.{} item: {}]'.format(i+1, des[:20]))
            valid = False
            ## the first line should never affect expense.
            for k,v in categories.items():
                if k in des:
                    print(des, '[', v, ']', seg4[i])
                    expense_cur[v] += float(seg4[i])
                    expense_tot[v] += float(seg4[i])
                    page_total += float(seg4[i])
                    valid_items += 1
                    valid = True
            if not valid:
                print('Considered invalid')
                unknown_spend[des] = seg4[i]
    
    month_total += page_total
    print('\t*** Total Spending in this page***')
    pprint(expense_cur)
    print('\t\tValid items in this page:', valid_items, '; Total in page: {}, New Spend:{}\n\n'.format(page_total, new_spend))

print('\t!!! Total Spending in this month !!! Total in this month: {}, New Spend:{}\n\n'.format(month_total, new_spend))
pprint(expense_tot)
print('\tUnknown Spend:\n', unknown_spend)
valid_amount = 50.0 ## minimum spending to be considered in the main category
valid_expense = {key: expense_tot[key] for key, val in expense_tot.items() if val >= valid_amount}
import matplotlib.pyplot as plt
print('create plots\n', valid_expense)
fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
# wedges, texts, autotexts =plt.pie(valid_expense.values(), frame=True, labels=valid_expense.keys(), autopct=lambda p : '{:.2f}%  ({:,.0f})'.format(p,p * sum(valid_expense.values())/100))
wedges, texts, autotexts =plt.pie(valid_expense.values(), labels=valid_expense.keys(), autopct=lambda p : '{:.2f}%  ({:,.0f})'.format(p,p * sum(valid_expense.values())/100))
# fig.legend(valid_expense.keys(), title="Category", loc="outside upper left")

plt.legend(valid_expense.keys(), title="Category", loc="center left", bbox_to_anchor=(-0.8, 0, 0.5, 1))
plt.setp(autotexts, size=6, weight="bold")
# plt.grid(True)
plt.title(title[0] + ': ' + str(month_total) + '(Calc), ' + str(new_spend)+'(Actual)')
plt.show()







