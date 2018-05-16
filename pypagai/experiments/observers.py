import csv
import logging
import os

from pypagai.util.model_persistence import ModelDumper
from sacred.observers.file_storage import FileStorageObserver

LOG = logging.getLogger('pypagai-logger')


class PypagAIFileStorageObserver(FileStorageObserver):

    def __init__(self, basedir, resource_dir=None, source_dir=None, template=None, priority=20):
        super(PypagAIFileStorageObserver, self).__init__(basedir, resource_dir, source_dir, template, priority)
        self.basedir = '/tmp/dummy-folder' if not basedir else basedir
        self.persis_model = False

    def write_file(self, name, dictionary):
        """
        Write dictionaries into csv file
        :param name:
        :param dictionary:
        :return:
        """
        for k, df in dictionary.items():
            df.to_csv(
                os.path.join(self.dir, '{}-{}.csv'.format(name, k)),
                quoting=csv.QUOTE_NONNUMERIC,
                index=False
            )

    def started_event(self, ex_info, command, host_info, start_time, config, meta_info, _id):
        """
        On this function we will create folder to storage experiments results
        """
        self.persis_model = not meta_info['options']['--unobserved']
        self.basedir = os.path.join(self.basedir, meta_info['options']['--name'])

        if not os.path.exists(self.basedir):
            os.makedirs(self.basedir)

        storage = super(PypagAIFileStorageObserver, self)
        return storage.started_event(ex_info, command, host_info, start_time, config, meta_info, _id)

    def heartbeat_event(self, info, captured_out, beat_time, result):

        for column in ['report', 'metrics']:
            if column in info and 'test' in info[column]:
                vis_frame = "\n\n{}\n\n##########################################\n"
                LOG.info(vis_frame.format(info[column]['test']))

        for column in ['raw_results', 'report', 'metrics']:
            if column in info:
                self.write_file(column, info[column])
                del info[column]

        storage = super(PypagAIFileStorageObserver, self)
        return storage.heartbeat_event(info, captured_out, beat_time, result)

    def completed_event(self, stop_time, result):

        LOG.info("[FINISH] experiment")

        storage = super(PypagAIFileStorageObserver, self)
        storage.completed_event(stop_time, '')

        if self.persis_model:
            ModelDumper(result).dump(os.path.join(self.dir, 'model.pkl'))
