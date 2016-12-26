import logging

import sys
import cv2
import numpy
import lxml

rootLogger = logging.getLogger()
rootLogger.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
rootLogger.addHandler(ch)

logging.debug(sys.argv)

files = sys.argv[1:]

hists = dict()
frame_times = dict()

for current_file in files:
    file_index = files.index(current_file)
    if file_index > 0:
        previous_index = file_index - 1
    else:
        previous_index = None

    logging.debug("processing %s, idx %s prev %s" % (current_file, file_index, previous_index))
    vid = cv2.VideoCapture(current_file)

    if not vid.isOpened():
        logging.error("couldn't open file")

    frame_times[file_index] = dict()
    hists[file_index] = dict()
    total_frame_count = vid.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)

    ret, frame = vid.read()
    while ret:
        cur_frame_idx = int(vid.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))
        cur_frame_pos_msec = long(vid.get(cv2.cv.CV_CAP_PROP_POS_MSEC))
        frame_height = int(vid.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
        frame_width = int(vid.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        logging.debug("current frame %s of %s, %sx%s" % (cur_frame_idx, total_frame_count, frame_width, frame_height))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = numpy.float32(gray)
        hist = cv2.calcHist(images=[gray],
                            channels=[0],
                            mask=None,
                            histSize=[256],
                            ranges=[0, 256])
        frame_times[file_index][cur_frame_idx] = cur_frame_pos_msec
        hists[file_index][cur_frame_idx] = hist

        #logging.debug(dst)

        ret, frame = vid.read()

    if previous_index is not None:
        current_file_hists = hists[file_index]
        prev_file_hists = hists[previous_index]

        current_frame_count = len(current_file_hists)
        prev_frame_count = len(prev_file_hists)

        logging.debug("%s features current file %s prev file" %(current_frame_count, prev_frame_count))

        correlations = dict()

        for frame_idx_prev_file, hist in current_file_hists.iteritems():
            similarity = 0
            try:
                first_frame_cur_file = 1
                similarity = cv2.compareHist(method=cv2.cv.CV_COMP_CORREL,
                                             H1=current_file_hists[first_frame_cur_file],
                                             H2=prev_file_hists[frame_idx_prev_file])
                frame_key = "%d,%d" % (frame_idx_prev_file, first_frame_cur_file)
                correlations[frame_key] = similarity
                if similarity > 0.98:
                    logging.debug("%s=%s" %(frame_key,similarity))
            except Exception as ex:
                ex=ex

        #logging.debug(correlations)
        best_match = max(correlations, key=correlations.get)
        logging.debug("best match %s=%s" % (best_match, correlations[best_match]))
    else:
        logging.debug("prev idx %s first file, not calculating hists" % previous_index)
