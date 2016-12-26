import logging
import sys

import cv2
import numpy

from FcpEvent import FcpEvent, SingleVideo


def compute_correlations(prev_file_hists, current_file_hists, frame_idx_from_previous):
    correlations = dict()
    for frame_index_current_file, hist in current_file_hists.iteritems():
        similarity = 0
        try:

            similarity = cv2.compareHist(method=cv2.cv.CV_COMP_CORREL,
                                         H1=hist,
                                         H2=prev_file_hists[frame_idx_from_previous])
            frame_key = frame_index_current_file
            correlations[frame_key] = similarity
            if similarity > 0.98:
                logging.debug("%s=%s" % (frame_key, similarity))
        except Exception as ex:
            logging.error(ex)

    return correlations


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
start_frames = dict()
frame_widths = dict()
frame_heights = dict()
end_frames = dict()

event = FcpEvent()

for file_index, current_file in enumerate(files):
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
    frame_heights[file_index] = int(vid.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    frame_widths[file_index] = int(vid.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    if ret:
        logging.debug("processing %s frames, %sx%s" % (total_frame_count, frame_widths[file_index], frame_heights[file_index]))
    while ret:
        cur_frame_idx = int(vid.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))
        cur_frame_pos_msec = long(vid.get(cv2.cv.CV_CAP_PROP_POS_MSEC))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = numpy.float32(gray)
        hist = cv2.calcHist(images=[gray],
                            channels=[0],
                            mask=None,
                            histSize=[256],
                            ranges=[0, 256])
        frame_times[file_index][cur_frame_idx] = cur_frame_pos_msec
        hists[file_index][cur_frame_idx] = hist

        # logging.debug(dst)

        ret, frame = vid.read()

    if previous_index is not None:
        current_file_hists = hists[file_index]
        prev_file_hists = hists[previous_index]

        current_frame_count = len(current_file_hists)
        prev_frame_count = len(prev_file_hists)

        logging.debug("%s features current file %s prev file" % (current_frame_count, prev_frame_count))

        max_frame_id_in_previous = max(frame_times[previous_index], key=frame_times[previous_index].get)
        max_frame_id = max(frame_times[file_index], key=frame_times[file_index].get)

        best_match = None
        best_match_score = 0
        best_previous_frame_idx = max_frame_id_in_previous
        found_good_match = False
        for frame_idx_from_previous in range(max_frame_id_in_previous, 0, -1):
            correlations = compute_correlations(prev_file_hists=prev_file_hists,
                                                current_file_hists=current_file_hists,
                                                frame_idx_from_previous=frame_idx_from_previous)
            current_best_match = max(correlations, key=correlations.get)

            if correlations[current_best_match] > best_match_score:
                logging.info("match between %s and %s frame %s in first %s in second is %s"
                             % (files[previous_index],
                                current_file,
                                frame_idx_from_previous,
                                current_best_match,
                                correlations[current_best_match]))
                best_match = current_best_match
                best_match_score = correlations[current_best_match]
                best_previous_frame_idx = frame_idx_from_previous
                if best_match_score > .999999:
                    found_good_match = True
                    break

        if found_good_match:
            logging.info(
                "found great match between %s and %s - best was between frame %s and %s with correlation %s"
                % (files[previous_index],
                   current_file,
                   best_previous_frame_idx,
                   best_match,
                   best_match_score)
                )
            end_frames[previous_index] = best_previous_frame_idx
            start_frames[file_index] = best_match
            end_frames[file_index] = max_frame_id
        else:
            logging.warn(
                "didn't find great match between %s and %s - best was between frame %s and %s with correlation %s"
                % (files[previous_index],
                   current_file,
                   best_previous_frame_idx,
                   best_match,
                   best_match_score)
                )
            end_frames[previous_index] = max_frame_id_in_previous
            start_frames[file_index] = 1
            end_frames[file_index] = max_frame_id

    else:
        logging.debug("prev idx %s first file, not calculating histogram similarities" % previous_index)

        max_frame_id = max(frame_times[file_index], key=frame_times[file_index].get)
        start_frames[file_index] = 1
        end_frames[file_index] = max_frame_id

for file_index, current_file in enumerate(files):
    start_frame = start_frames[file_index]
    end_frame = end_frames[file_index]
    fcp_video = SingleVideo(filename=current_file,
                            start_msec=frame_times[file_index][start_frame] - frame_times[file_index][1],
                            end_msec=frame_times[file_index][end_frame] - frame_times[file_index][1],
                            frame_width=frame_widths[file_index],
                            frame_height=frame_heights[file_index])
    logging.debug(fcp_video)
    event.append_video(fcp_video)

with open('/tmp/findframes.fcpxml', 'w') as f:
    f.write(event.to_xml())
