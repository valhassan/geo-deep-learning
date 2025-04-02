import gc
import csv
import logging
import multiprocessing
import shutil
from datetime import datetime
from pathlib import Path
from numbers import Number
from typing import Sequence, Union

import rasterio
from matplotlib import pyplot as plt
from omegaconf import open_dict, DictConfig
from rasterio.plot import show_hist, show
from tqdm import tqdm

from dataset.aoi import AOI
from utils.aoiutils import aois_from_csv
from utils.utils import read_csv, get_key_def, get_git_hash, map_wrapper


def verify_per_aoi(
        aoi: AOI,
        output_report_dir: Union[str, Path],
        extended_label_stats: bool = True,
        output_raster_stats: bool = True,
        output_raster_plots: bool = True
):
    """
    Verifies a single AOI
    @param aoi:
        AOI object containing raster and label data to verify
    @param extended_label_stats:
        if True, will calculate polygon-related stats on label (mean area, mean perimeter, mean number of vertices)
    @param output_raster_stats:
        if True, will output stats on raster radiometric data
    @param output_raster_plots:
        if True, will output plots of RGB raster and histogram for all bands
    @param output_report_dir:
        Path where output report as csv should be written.
    @return:
        Returns info on AOI or error raised, if any.
    """
    try:
        if not aoi.raster or aoi.raster_closed:
            try:
                aoi.raster = rasterio.open(aoi.raster_dest)
                aoi.raster_closed = False
            except Exception as e:
                logging.error(f"Error opening raster for {aoi.aoi_id}: {e}")
                return None, e

        # get aoi info
        logging.info(f"\nGetting data info for {aoi.aoi_id}...")
        try:
            aoi_dict = aoi.to_dict(extended=extended_label_stats)
        except Exception as e:
            logging.error(f"Error getting aoi info for {aoi.aoi_id}: {e}")
            aoi_dict = {'id': aoi.aoi_id, 'error': str(e)}

        # Check that `num_classes` is equal to number of classes detected in the specified attribute for each GeoPackage
        if aoi.attr_field_filter and hasattr(aoi, 'label_gdf_filtered') and aoi.label_gdf_filtered is not None:
            try:
                label_unique_classes = aoi.label_gdf_filtered[aoi.attr_field_filter].unique()
                aoi_dict['label_unique_classes'] = label_unique_classes
            except Exception:
                aoi_dict['label_unique_classes'] = None
        
        if output_raster_stats:
            logging.info(f"\nGetting raster stats for {aoi.aoi_id}...")
            try:
                aoi_stats = aoi.calc_raster_stats()  # creates self.raster_np
                for cname, stats in aoi_stats.items():
                    aoi_dict.update(
                        {f"{cname}_{stat_name}": stat_val for stat_name, stat_val in stats['statistics'].items()})
                # aoi_dict.update({f"{cname}_buckets": stats['histogram']['buckets']})
            except Exception as e:
                logging.error(f"Error getting raster stats for {aoi.aoi_id}: {e}")
                aoi_dict['stats_error'] = str(e)

        if output_raster_plots:
            try:
                logging.info(f"\nGenerating plots for {aoi.aoi_id}...")
                out_plot = Path(output_report_dir) / f"raster_{aoi.aoi_id}.png"
                # https://rasterio.readthedocs.io/en/latest/topics/plotting.html
                fig, (axrgb, axhist) = plt.subplots(1, 2, figsize=(14, 7))
                aoi.raster_np = aoi.raster.read() if aoi.raster_np is None else aoi.raster_np  # prevent read if in memory
                show(aoi.raster_np, ax=axrgb, transform=aoi.raster.transform)
                show_hist(
                    aoi.raster_np, bins=50, lw=1.0, stacked=False, alpha=0.75,
                    histtype='step', title="Histogram", ax=axhist, label=aoi.raster_bands_request)
                plt.title(aoi.aoi_id)
                plt.savefig(out_plot)
                logging.info(f"Saved plot: {out_plot}")
                plt.close()
            except Exception as e:
                logging.error(f"Error generating plots for {aoi.aoi_id}: {e}")
                aoi_dict['plot_error'] = str(e)
        
        if aoi.raster and not aoi.raster_closed:
            aoi.close_raster()
        
        if hasattr(aoi, 'raster_np') and aoi.raster_np is not None:
            aoi.raster_np = None
        gc.collect()
        
        return aoi_dict, None
    except Exception as e:
        if hasattr(aoi, 'raster') and aoi.raster and not aoi.raster_closed:
            try:
                aoi.close_raster()
            except:
                pass
        gc.collect()
        return None, e

def main(cfg: DictConfig) -> None:
    """
    Data verification pipeline with batch processing to manage memory usage.
    Each batch of AOIs is created, verified, and written to output before moving to the next batch.
    """
    # PARAMETERS
    num_classes = len(cfg.dataset.classes_dict.keys())
    bands_requested = get_key_def('bands', cfg['dataset'], default=[], expected_type=Sequence)
    csv_file = get_key_def('raw_data_csv', cfg['dataset'], to_path=True, validate_path_exists=True)
    data_dir = get_key_def('raw_data_dir', cfg['dataset'], default="data", to_path=True, validate_path_exists=True)
    download_data = get_key_def('download_data', cfg['dataset'], default=False, expected_type=bool)

    dontcare = cfg.dataset.ignore_index if cfg.dataset.ignore_index is not None else -1
    if dontcare == 0:
        raise ValueError("\nThe 'dontcare' value (or 'ignore_index') used in the loss function cannot be zero.")
    attribute_field = get_key_def('attribute_field', cfg['dataset'], None)
    attr_vals = get_key_def('attribute_values', cfg['dataset'], None, expected_type=(Sequence, int))

    output_report_dir = get_key_def('output_report_dir', cfg['verify'], to_path=True, validate_path_exists=True)
    output_raster_stats = get_key_def('output_raster_stats', cfg['verify'], default=False, expected_type=bool)
    output_raster_plots = get_key_def('output_raster_plots', cfg['verify'], default=False, expected_type=bool)
    extended_label_stats = get_key_def('extended_label_stats', cfg['verify'], default=False, expected_type=bool)
    parallel = get_key_def('multiprocessing', cfg['verify'], default=False, expected_type=bool)
    write_dest_raster = get_key_def('write_dest_raster', cfg['verify'], default=False, expected_type=bool)
    clahe_clip_limit = get_key_def('clahe_clip_limit', cfg['tiling'], expected_type=Number, default=0)

    # ADD GIT HASH FROM CURRENT COMMIT TO PARAMETERS
    with open_dict(cfg):
        cfg.general.git_hash = get_git_hash()

    # Read the full CSV once
    data_list = read_csv(csv_file)
    logging.info(f'\n\tSuccessfully read csv file: {Path(csv_file).name}\n'
                 f'\tNumber of rows: {len(data_list)}\n'
                 f'\tCopying first row:\n{data_list[0]}\n')

    # Define batch size for processing
    batch_size = 2  # Adjust based on your dataset and memory constraints
    
    # Create output files
    outpath_csv = output_report_dir / f"report_info_{csv_file.stem}.csv"
    outpath_csv_errors = output_report_dir / f"report_error_{csv_file.stem}.log"
    
    # Initialize CSV writer for the first batch
    header_written = False
    errors = []
    
    # Process in batches
    for batch_idx in range(0, len(data_list), batch_size):
        batch_start = batch_idx
        batch_end = min(batch_idx + batch_size, len(data_list))
        batch_data = data_list[batch_start:batch_end]
        
        logging.info(f"\nProcessing batch {batch_idx//batch_size + 1} of {(len(data_list)-1)//batch_size + 1} (rows {batch_start+1}-{batch_end})")
        
        # STEP 1: Create AOIs for this batch
        batch_aois = []
        with tqdm(enumerate(batch_data), desc=f"Creating AOIs (batch {batch_idx//batch_size + 1})", 
                  total=len(batch_data)) as _tqdm:
            for i, aoi_dict in _tqdm:
                actual_idx = i + batch_start
                _tqdm.set_postfix_str(f"Image: {Path(aoi_dict['tif']).stem}")
                try:
                    new_aoi = AOI.from_dict(
                        aoi_dict=aoi_dict,
                        bands_requested=bands_requested,
                        attr_field_filter=attribute_field,
                        attr_values_filter=attr_vals,
                        download_data=download_data,
                        root_dir=data_dir,
                        for_multiprocessing=parallel,
                        write_dest_raster=write_dest_raster,
                        equalize_clahe_clip_limit=clahe_clip_limit,
                    )
                    
                    # Immediately close raster file
                    if hasattr(new_aoi, 'raster') and new_aoi.raster is not None:
                        new_aoi.close_raster()
                        new_aoi.raster = None
                    
                    batch_aois.append(new_aoi)
                    
                    # Force garbage collection after each AOI
                    del new_aoi
                    gc.collect()
                    
                except Exception as e:
                    err_msg = f"Error creating AOI for row {actual_idx}: {str(e)}\n"
                    logging.error(err_msg)
                    errors.append(err_msg)
        
        # STEP 2: Verify AOIs for this batch
        batch_reports = []
        with tqdm(batch_aois, desc=f"Verifying AOIs (batch {batch_idx//batch_size + 1})", total=len(batch_aois)) as _tqdm:
            for aoi in _tqdm:
                _tqdm.set_postfix_str(f"AOI: {aoi.aoi_id}")
                
                if parallel and len(batch_aois) > 1:
                    # Set up multiprocessing for this batch only
                    input_args = [[verify_per_aoi, aoi, output_report_dir, extended_label_stats,
                                   output_raster_stats, output_raster_plots]]
                    
                    with multiprocessing.get_context('spawn').Pool(processes=1) as pool:
                        results = pool.map_async(map_wrapper, input_args).get()
                        
                    aoi_dict, error = results[0]
                else:
                    aoi_dict, error = verify_per_aoi(aoi, output_report_dir, extended_label_stats,
                                                     output_raster_stats, output_raster_plots)
                
                if aoi_dict:
                    batch_reports.append(aoi_dict)
                if error:
                    err_msg = f"Error verifying AOI {aoi.aoi_id}: {str(error)}\n"
                    logging.error(err_msg)
                    errors.append(err_msg)
                
                # Clean up
                del aoi
                gc.collect()
        
        # STEP 3: Write results for this batch
        if batch_reports:
            write_mode = 'a' if header_written else 'w'
            with open(outpath_csv, write_mode, newline='') as output_file:
                if not header_written and batch_reports:
                    dict_writer = csv.DictWriter(output_file, batch_reports[0].keys())
                    dict_writer.writeheader()
                    header_written = True
                else:
                    dict_writer = csv.DictWriter(output_file, batch_reports[0].keys())
                
                dict_writer.writerows(batch_reports)
            
            logging.info(f"Wrote {len(batch_reports)} AOI reports for batch {batch_idx//batch_size + 1}")
        
        # Clear batch data and force garbage collection
        batch_aois = []
        batch_reports = []
        gc.collect()
        
        logging.info(f"Completed batch {batch_idx//batch_size + 1} of {(len(data_list)-1)//batch_size + 1}")
    
    # Write any errors at the end
    if errors:
        with open(outpath_csv_errors, 'w') as error_file:
            error_file.writelines(errors)
        logging.warning(f"Verification completed with {len(errors)} errors. See {outpath_csv_errors} for details.")
    else:
        logging.info(f"Verification completed successfully with no errors.")
    
    logging.info(f"\nInput data verification done. See outputs in {output_report_dir}")