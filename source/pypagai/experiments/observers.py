import csv
import os

from pypagai.util.model_persistence import ModelDumper
from sacred.observers.file_storage import FileStorageObserver


class PypagAIFileStorageObserver(FileStorageObserver):

    def __init__(self):
        self.basedir = '/tmp/dummy-folder'
        self.persis_model = False

    def started_event(self, ex_info, command, host_info, start_time, config, meta_info, _id):
        """
        On this function we will create folder to storage experiments results
        """

        self.basedir = os.path.join(self.basedir, meta_info['options']['--name'])

        if not os.path.exists(self.basedir):
            os.makedirs(self.basedir)

        storage = super(PypagAIFileStorageObserver, self)
        return storage.started_event(ex_info, command, host_info, start_time, config, meta_info, _id)

    def heartbeat_event(self, info, captured_out, beat_time, result):

        print(beat_time)
        print(result)

        if 'raw_results' in info:

            results = info['raw_results']

            for k, df in results.items():
                df.to_csv(
                    os.path.join(self.dir, 'raw_results_{}.csv'.format(k)),
                    quoting=csv.QUOTE_NONNUMERIC,
                    index=False
                )

            del info['raw_results']

        storage = super(PypagAIFileStorageObserver, self)
        return storage.heartbeat_event(info, captured_out, beat_time, result)

    def completed_event(self, stop_time, result):
        storage = super(PypagAIFileStorageObserver, self)
        storage.completed_event(stop_time, '')

        if self.persis_model:
            ModelDumper(result).dump(os.path.join(self.dir, 'model.pkl'))


            # r = {
            #     'model': model.__name__,
            #     'acc': acc,
            #     'f1': f1,
            #     'db': db_cfg['reader'].ALIAS,
            #     'db_parameters': json.dumps(
            #         {k: v if isinstance(v, str) or isinstance(v, int) or isinstance(v, float) else v.__name__ for
            #          k, v in dataset_cfg.items()}),
            #     'model_cfg': json.dumps(
            #         {k: v if isinstance(v, str) or isinstance(v, int) or isinstance(v, float) else v.__name__ for
            #          k, v in dataset_cfg.items()}),
            # }
            #
            # results.append(r)
            # df = pd.DataFrame(results)
            # df.to_csv('result.csv', sep=';', index=False)
