
import os
import logging
import datetime

from pathlib import Path

from common.terminal import colorstr


class HardLogger(logging.Logger):
    def __init__(self, project_path: str = '../data/demo/experiments', name: str = None, export_data_path: Path = None):

        self.datetime_tag = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        self.export_data_path = Path(
            export_data_path) if export_data_path is not None else Path('logs')
        self.name = Path(
            name + '_' + self.datetime_tag) if name is not None else Path(self.datetime_tag)
        self.logger = logging.getLogger(__name__)

        self.parent_dir = self.export_data_path / self.name
        self.project_path = os.path.abspath(
            os.path.join(__file__, project_path))

        self.parent_dir_printable_version = str(os.path.abspath(
            self.parent_dir)).replace(':', '').replace('/', ' > ')
        self.project_path_printable_version = str(
            self.project_path).replace(':', '').replace('/', ' > ')

        self.model_dir = self.export_data_path / self.name / "model"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.data_dir = self.export_data_path / self.name / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = self.export_data_path / self.name / "log"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        try:
            f = open(self.log_dir / "logger.log", "x")
            f.close()
        except:
            raise PermissionError(
                f"Could not create the file {self.log_dir / 'logger.log'}.")

        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            datefmt='%Y-%m-%dT%H-%M-%S', filename=self.log_dir / "logger.log", filemode='w')

    def log_stats(self, acc: float, loss: float, epoch: int, desc: str = 'Testing'):
        self.logger.info(
            f"[Process: {desc:>15}] "
            f"[Epoch: {epoch:05d}] "
            f"[Loss: {loss:7.3f}] "
            f"[Accuracy: {acc:7.3f}]")

    def print_training_message(
        self,
        model_id: str,
        dataset_id: str,
        epochs: int,
        device: str,
        elite_metric: str,
        epoch: int,
        resume: str,
        auto_save: bool
    ):
        new_line = '\n'
        tab_char = '\t'

        print(f"\n\n\t\t    Training a {colorstr(options=['red', 'underline'], string_args=list([model_id]))} on {colorstr(options=['red', 'underline'], string_args=list([dataset_id]))} for {colorstr(options=['red', 'underline'], string_args=list([str(epochs)]))} epochs using\n"
              f"\t\t  a {colorstr(options=['red', 'underline'], string_args=list([device.upper()])) + ' enabled device' if 'cuda' in device.lower() else colorstr(options=['red', 'underline'], string_args=list([device.upper()]))}. {colorstr(options=['blue', 'bold'], string_args=list(['Simulator']))} will select checkpoints \n"
              f"\t\t      based on {colorstr(options=['red', 'underline'], string_args=list([elite_metric]))} {'accuracy' if 'top' in elite_metric else 'metric'}. Auto-saving is {colorstr(options=['blue'], string_args=list(['enabled'])) if auto_save else colorstr(options=['blue'], string_args=list(['disabled']))}{' and' + new_line + tab_char + tab_char + '         the training will start from epoch ' + colorstr(options=['red', 'underline'], string_args=list([str(epoch)])) + '.' + new_line if resume else '.' + new_line}"
              f"\n\n\t      The experiment logger is uploaded locally at: \n"
              f"  {colorstr(options=['blue', 'underline'], string_args=list([self.parent_dir_printable_version]))}."
              f"\n\n\t\t              Project absolute path is:\n\t"
              f"{colorstr(options=['blue', 'underline'], string_args=list([self.project_path_printable_version]))}.\n\n")

    def print_test_message(self, model_id: str, dataset_id: str, epochs: int, device: str):
        print(f"\n\n\t\t    Evaluating a {colorstr(options=['red', 'underline'], string_args=list([model_id]))} on {colorstr(options=['red', 'underline'], string_args=list([dataset_id]))} for {colorstr(options=['red', 'underline'], string_args=list([str(epochs)]))} epochs using "
              f"a {colorstr(options=['red', 'underline'], string_args=list([device.upper()])) + ' enabled device' if 'cuda' in device.lower() else colorstr(options=['red', 'underline'], string_args=list([device.upper()]))}."
              f"\n\n\t      The experiment logger is uploaded locally at: \n"
              f"  {colorstr(options=['blue', 'underline'], string_args=list([self.parent_dir_printable_version]))}."
              f"\n\n\t\t              Project absolute path is:\n\t"
              f"{colorstr(options=['blue', 'underline'], string_args=list([self.project_path_printable_version]))}.\n\n")
