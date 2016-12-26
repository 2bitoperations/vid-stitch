import urllib

from lxml import etree


class SingleVideo:
    def __init__(self, filename, start_msec, end_msec, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.start_msec = start_msec
        self.filename = filename
        self.end_msec = end_msec

    def __str__(self):
        return "%s %sx%s, start %s end %s" % (self.filename,
                                              self.frame_width,
                                              self.frame_height,
                                              self.start_msec,
                                              self.end_msec)


class FcpEvent:
    def __init__(self):
        self.videos = []

    def append_video(self, video):
        self.videos.append(video)

    def to_xml(self):
        root = etree.Element("fcpxml", version="1.6")

        resources = etree.Element("resources")

        format_name_1080p_30 = "f1080p"
        format_name_720p_30 = "f720p"
        format_1080p_30 = etree.Element("format", id=format_name_1080p_30, name="FFVideoFormat1080p30")
        format_720p_30 = etree.Element("format", id=format_name_720p_30, name="FFVideoFormat720p30")
        resources.append(format_1080p_30)
        resources.append(format_720p_30)

        formats_by_height = dict()
        formats_by_height[1080] = format_name_1080p_30
        formats_by_height[720] = format_name_720p_30

        event = etree.Element("event", name="JoinedVideos")

        media_for_sequence_id = "media0"
        media_for_sequence = etree.Element("media",
                                           id=media_for_sequence_id,
                                           name="CompoundClip-%s" % media_for_sequence_id)
        sequence = etree.Element("sequence",
                                 duration="0s",
                                 format=format_name_1080p_30,
                                 renderColorSpace="Rec. 709",
                                 tcStart="0s",
                                 tcFormat="DF"
                                 )
        spine = etree.Element("spine")

        running_offset = 0.0
        for i, video in enumerate(self.videos):
            asset_ref = "asset%s" % i
            clip_ref = "clip%s" % i
            format_ref = formats_by_height[video.frame_height]
            duration_secs = (video.end_msec - video.start_msec) / 1000
            duration_secs_formatted = "%fs" % duration_secs
            start_secs_formatted = "%fs" % (video.start_msec / 1000)
            asset = etree.Element("asset",
                                  id=asset_ref,
                                  src="file://%s" % urllib.quote(video.filename),
                                  start=start_secs_formatted,
                                  duration=duration_secs_formatted,
                                  hasVideo="1",
                                  hasAudio="1",
                                  format=format_ref,
                                  audioSources="1",
                                  audioChannels="2",
                                  audioRate="32000"
                                  )
            resources.append(asset)

            asset_clip = etree.Element("asset-clip",
                                       name=clip_ref,
                                       ref=asset_ref,
                                       format=format_ref,
                                       start="0s",
                                       duration=duration_secs_formatted,
                                       audioRole="dialogue"
                                       )

            event.append(asset_clip)

            asset_clip_for_spine = etree.Element("asset-clip",
                                                 name=clip_ref,
                                                 offset="%fs" % running_offset,
                                                 start=start_secs_formatted,
                                                 duration=duration_secs_formatted,
                                                 ref=asset_ref,
                                                 audioRole="dialogue",
                                                 tcFormat="DF")

            spine.append(asset_clip_for_spine)

            running_offset += duration_secs

        ref_clip = etree.Element("ref-clip",
                                 name="JoinedClips",
                                 ref=media_for_sequence_id,
                                 duration="%fs" % running_offset)
        event.append(ref_clip)

        sequence.append(spine)
        media_for_sequence.append(sequence)
        resources.append(media_for_sequence)

        root.append(resources)
        root.append(event)

        return etree.tostring(root, pretty_print=True)
