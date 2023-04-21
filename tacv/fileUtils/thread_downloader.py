"""This code was taken and modified from https://gist.github.com/chandlerprall/1017266"""

import threading
import sys, os, re
import time
from queue import Queue
from urllib.request import urlopen

from attr.validators import instance_of


class ThreadedDownload(object):
    REGEX = {
        'hostname_strip': re.compile('.*\..*?/', re.I)
    }

    class MissingDirectoryException(Exception):
        pass

    class DownloadTracker(threading.Thread):
        def __init__(self, queue):
            super(ThreadedDownload.DownloadTracker, self).__init__()
            self.queue = queue

        def run(self) -> None:
            start_time = time.time()
            queue_start_size = max(1, self.queue.qsize())
            while not self.queue.empty():
                queue_size = self.queue.qsize()
                done_percent = 100 * ((queue_start_size - queue_size) / queue_start_size)
                taken_time = time.time() - start_time
                if done_percent > 0:
                    remaining = (100 - done_percent) * taken_time / done_percent
                    print(f"\rDone {done_percent}%, {int(remaining)} seconds remaining.")
                time.sleep(0.5)

    class Downloader(threading.Thread):
        def __init__(self, queue: Queue, report):
            threading.Thread.__init__(self)
            self.queue = queue
            self.report = report

        def run(self):
            while self.queue.empty() == False:
                url = self.queue.get()

                response = url.download()
                if response == False and url.url_tried < url.url_tries:
                    self.queue.put(url)
                elif response == False and url.url_tried == url.url_tries:
                    self.report['failure'].append(url.url)
                elif response == True:
                    self.report['success'].append(url.url)

                self.queue.task_done()

    class URLTarget(object):
        def __init__(self, url, destination, url_tries):
            self.url = url
            self.destination = destination
            self.url_tries = url_tries
            self.url_tried = 0
            self.success = False
            self.error = None

        def download(self):
            self.url_tried = self.url_tried + 1

            try:
                if os.path.exists(self.destination):  # This file has already been downloaded
                    self.success = True
                    return self.success

                remote_file = urlopen(self.url)
                package = remote_file.read()
                remote_file.close()

                if os.path.exists(os.path.dirname(self.destination)) == False:
                    os.makedirs(os.path.dirname(self.destination))

                dest_file = open(self.destination, 'wb')
                dest_file.write(package)
                dest_file.close()

                self.success = True

            except Exception as e:
                self.error = e

            return self.success

        def __str__(self):
            return 'URLTarget (%(url)s, %(success)s, %(error)s)' % {'url': self.url, 'success': self.success,
                                                                    'error': self.error}

    def __init__(self, urls=[], destination='.', directory_structure=False, thread_count=5, url_tries=3):
        if os.path.exists(destination) is False:
            os.makedirs(destination)

        self.queue = Queue(0)  # Infinite sized queue
        self.report = {'success': [], 'failure': []}
        self.threads = []

        if destination[-1] != os.path.sep:
            destination = destination + os.path.sep
        self.destination = destination
        self.thread_count = thread_count
        self.directory_structure = directory_structure

        # Prepopulate queue with any values we were given
        for url in urls:
            self.queue.put(ThreadedDownload.URLTarget(url, self.fileDestination(url), url_tries))

    def fileDestination(self, url):
        if self.directory_structure == False:
            # No directory structure, just filenames
            file_destination = '%s%s' % (self.destination, os.path.basename(url))

        elif self.directory_structure == True:
            # Strip off hostname, keep all other directories
            file_destination = '%s%s' % (self.destination, ThreadedDownload.REGEX['hostname_strip'].sub('', url))

        elif hasattr(self.directory_structure, '__len__') and len(self.directory_structure) == 2:
            # User supplied a custom regex replace
            regex = self.directory_structure[0]
            if instance_of(regex, str):
                regex = re.compile(str)
            replace = self.directory_structure[1]
            file_destination = '%s%s' % (self.destination, regex.sub(replace, url))

        else:
            # No idea what's wanted
            file_destination = None

        if hasattr(file_destination, 'replace'):
            file_destination = file_destination.replace('/', os.path.sep)
        return file_destination

    def addTarget(self, url, url_tries=3):
        self.queue.put(ThreadedDownload.URLTarget(url, self.fileDestination(url), url_tries))

    def run(self):
        for i in range(self.thread_count):
            thread = ThreadedDownload.Downloader(self.queue, self.report)
            thread.start()
            self.threads.append(thread)
        if self.queue.qsize() > 0:
            download_tracker = ThreadedDownload.DownloadTracker(self.queue)
            download_tracker.run()
            self.queue.join()
