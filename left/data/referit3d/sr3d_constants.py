#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : sr3d_constants.py
# Author : Joy Hsu
# Email  : joycj@stanford.edu
# Date   : 10/04/2023
#
# This file is part of Project Left.
# Distributed under terms of the MIT license.

attribute_concepts = ['air_hockey_table_Object',
 'air_mattress_Object',
 'airplane_Object',
 'alarm_Object',
 'alarm_clock_Object',
 'armchair_Object',
 'baby_changing_station_Object',
 'baby_mobile_Object',
 'backpack_Object',
 'bag_Object',
 'bag_of_coffee_beans_Object',
 'ball_Object',
 'banana_holder_Object',
 'bananas_Object',
 'banister_Object',
 'banner_Object',
 'bar_Object',
 'barricade_Object',
 'barrier_Object',
 'baseball_cap_Object',
 'basket_Object',
 'bath_products_Object',
 'bath_walls_Object',
 'bathrobe_Object',
 'bathroom_cabinet_Object',
 'bathroom_counter_Object',
 'bathroom_stall_Object',
 'bathroom_stall_door_Object',
 'bathroom_vanity_Object',
 'bathtub_Object',
 'battery_disposal_jar_Object',
 'beachball_Object',
 'beanbag_chair_Object',
 'bear_Object',
 'bed_Object',
 'bedframe_Object',
 'beer_bottles_Object',
 'bench_Object',
 'bicycle_Object',
 'bike_lock_Object',
 'bike_pump_Object',
 'bin_Object',
 'binder_Object',
 'binders_Object',
 'blackboard_Object',
 'blackboard_eraser_Object',
 'blanket_Object',
 'blinds_Object',
 'block_Object',
 'board_Object',
 'boards_Object',
 'boat_Object',
 'boiler_Object',
 'book_Object',
 'book_rack_Object',
 'books_Object',
 'bookshelf_Object',
 'bookshelves_Object',
 'boots_Object',
 'bottle_Object',
 'bottles_Object',
 'bowl_Object',
 'box_Object',
 'boxes_Object',
 'boxes_of_paper_Object',
 'breakfast_bar_Object',
 'briefcase_Object',
 'broom_Object',
 'bucket_Object',
 'buddha_Object',
 'bulletin_board_Object',
 'bunk_bed_Object',
 'bycicle_Object',
 'cabinet_Object',
 'cabinet_door_Object',
 'cabinet_doors_Object',
 'cabinets_Object',
 'cable_Object',
 'calendar_Object',
 'camera_Object',
 'can_Object',
 'candle_Object',
 'canopy_Object',
 'cap_Object',
 'car_Object',
 'card_Object',
 'cardboard_Object',
 'carpet_Object',
 'carseat_Object',
 'cart_Object',
 'carton_Object',
 'case_Object',
 'case_of_water_bottles_Object',
 'cat_litter_box_Object',
 'cd_case_Object',
 'cd_cases_Object',
 'ceiling_Object',
 'ceiling_fan_Object',
 'ceiling_lamp_Object',
 'ceiling_light_Object',
 'centerpiece_Object',
 'chain_Object',
 'chair_Object',
 'chandelier_Object',
 'changing_station_Object',
 'chest_Object',
 'clip_Object',
 'clock_Object',
 'closet_Object',
 'closet_ceiling_Object',
 'closet_door_Object',
 'closet_doorframe_Object',
 'closet_doors_Object',
 'closet_floor_Object',
 'closet_rod_Object',
 'closet_shelf_Object',
 'closet_wall_Object',
 'closet_walls_Object',
 'closet_wardrobe_Object',
 'cloth_Object',
 'clothes_Object',
 'clothes_dryer_Object',
 'clothes_dryers_Object',
 'clothes_hanger_Object',
 'clothes_hangers_Object',
 'clothing_Object',
 'clothing_rack_Object',
 'coat_Object',
 'coat_rack_Object',
 'coatrack_Object',
 'coffee_bean_bag_Object',
 'coffee_box_Object',
 'coffee_kettle_Object',
 'coffee_maker_Object',
 'coffee_mug_Object',
 'coffee_table_Object',
 'column_Object',
 'compost_bin_Object',
 'computer_tower_Object',
 'conditioner_bottle_Object',
 'cone_Object',
 'contact_lens_solution_bottle_Object',
 'container_Object',
 'controller_Object',
 'cooking_pan_Object',
 'cooking_pot_Object',
 'cooler_Object',
 'copier_Object',
 'cosmetic_bag_Object',
 'costume_Object',
 'couch_Object',
 'couch_cushions_Object',
 'counter_Object',
 'cover_Object',
 'covered_box_Object',
 'crate_Object',
 'crib_Object',
 'crutches_Object',
 'cup_Object',
 'cups_Object',
 'curtain_Object',
 'curtain_rod_Object',
 'curtains_Object',
 'cushion_Object',
 'cutting_board_Object',
 'dart_board_Object',
 'decoration_Object',
 'desk_Object',
 'desk_lamp_Object',
 'diaper_bin_Object',
 'dining_table_Object',
 'dish_rack_Object',
 'dishwasher_Object',
 'dishwashing_soap_bottle_Object',
 'dispenser_Object',
 'display_Object',
 'display_case_Object',
 'display_rack_Object',
 'display_sign_Object',
 'divider_Object',
 'doll_Object',
 'dollhouse_Object',
 'dolly_Object',
 'door_Object',
 'door_wall_Object',
 'doorframe_Object',
 'doors_Object',
 'drawer_Object',
 'dress_rack_Object',
 'dresser_Object',
 'drum_set_Object',
 'dryer_sheets_Object',
 'drying_rack_Object',
 'duffel_bag_Object',
 'dumbbell_Object',
 'dumbbell_plates_Object',
 'dumbell_Object',
 'dustpan_Object',
 'easel_Object',
 'electric_panel_Object',
 'elevator_Object',
 'elevator_button_Object',
 'elliptical_machine_Object',
 'end_table_Object',
 'envelope_Object',
 'exercise_ball_Object',
 'exercise_bike_Object',
 'exercise_machine_Object',
 'exit_sign_Object',
 'fan_Object',
 'faucet_Object',
 'file_cabinet_Object',
 'file_cabinets_Object',
 'file_organizer_Object',
 'film_light_Object',
 'fire_alarm_Object',
 'fire_extinguisher_Object',
 'fire_hose_Object',
 'fire_sprinkler_Object',
 'fireplace_Object',
 'fish_Object',
 'flag_Object',
 'flip_flops_Object',
 'floor_Object',
 'flower_stand_Object',
 'flowerpot_Object',
 'folded_boxes_Object',
 'folded_chair_Object',
 'folded_chairs_Object',
 'folded_ladder_Object',
 'folded_table_Object',
 'folder_Object',
 'food_bag_Object',
 'food_container_Object',
 'food_display_Object',
 'foosball_table_Object',
 'footrest_Object',
 'footstool_Object',
 'frame_Object',
 'frying_pan_Object',
 'furnace_Object',
 'furniture_Object',
 'fuse_box_Object',
 'futon_Object',
 'gaming_wheel_Object',
 'garage_door_Object',
 'garbage_bag_Object',
 'glass_Object',
 'glass_doors_Object',
 'globe_Object',
 'golf_bag_Object',
 'grab_bar_Object',
 'grocery_bag_Object',
 'guitar_Object',
 'guitar_case_Object',
 'hair_brush_Object',
 'hair_dryer_Object',
 'hamper_Object',
 'hand_dryer_Object',
 'hand_rail_Object',
 'hand_sanitzer_dispenser_Object',
 'hand_towel_Object',
 'handicap_bar_Object',
 'handrail_Object',
 'hanging_Object',
 'hat_Object',
 'hatrack_Object',
 'headboard_Object',
 'headphones_Object',
 'heater_Object',
 'helmet_Object',
 'hose_Object',
 'hoverboard_Object',
 'humidifier_Object',
 'ikea_bag_Object',
 'instrument_case_Object',
 'ipad_Object',
 'iron_Object',
 'ironing_board_Object',
 'jacket_Object',
 'jar_Object',
 'jewelry_box_Object',
 'kettle_Object',
 'keyboard_Object',
 'keyboard_piano_Object',
 'kinect_Object',
 'kitchen_apron_Object',
 'kitchen_cabinet_Object',
 'kitchen_cabinets_Object',
 'kitchen_counter_Object',
 'kitchen_island_Object',
 'kitchen_mixer_Object',
 'kitchenaid_mixer_Object',
 'knife_block_Object',
 'ladder_Object',
 'lamp_Object',
 'lamp_base_Object',
 'laptop_Object',
 'laundry_bag_Object',
 'laundry_basket_Object',
 'laundry_detergent_Object',
 'laundry_hamper_Object',
 'ledge_Object',
 'legs_Object',
 'light_Object',
 'light_switch_Object',
 'loft_bed_Object',
 'loofa_Object',
 'lotion_Object',
 'lotion_bottle_Object',
 'luggage_Object',
 'luggage_rack_Object',
 'luggage_stand_Object',
 'lunch_box_Object',
 'machine_Object',
 'magazine_Object',
 'magazine_rack_Object',
 'mail_Object',
 'mail_bin_Object',
 'mail_tray_Object',
 'mail_trays_Object',
 'mailbox_Object',
 'mailboxes_Object',
 'map_Object',
 'massage_chair_Object',
 'mat_Object',
 'mattress_Object',
 'medal_Object',
 'media_center_Object',
 'messenger_bag_Object',
 'metronome_Object',
 'microwave_Object',
 'mini_fridge_Object',
 'mirror_Object',
 'mirror_doors_Object',
 'monitor_Object',
 'mop_Object',
 'mouse_Object',
 'mouthwash_bottle_Object',
 'mug_Object',
 'music_book_Object',
 'music_stand_Object',
 'nerf_gun_Object',
 'night_lamp_Object',
 'night_light_Object',
 'nightstand_Object',
 'notepad_Object',
 'object_Object',
 'office_chair_Object',
 'open_kitchen_cabinet_Object',
 'organizer_Object',
 'organizer_shelf_Object',
 'ottoman_Object',
 'oven_Object',
 'oven_mitt_Object',
 'painting_Object',
 'pantry_shelf_Object',
 'pantry_wall_Object',
 'pantry_walls_Object',
 'pants_Object',
 'paper_Object',
 'paper_bag_Object',
 'paper_cutter_Object',
 'paper_organizer_Object',
 'paper_shredder_Object',
 'paper_towel_Object',
 'paper_towel_dispenser_Object',
 'paper_towel_roll_Object',
 'paper_towel_rolls_Object',
 'paper_tray_Object',
 'papers_Object',
 'pen_holder_Object',
 'person_Object',
 'photo_Object',
 'piano_Object',
 'piano_bench_Object',
 'picture_Object',
 'pictures_Object',
 'pillar_Object',
 'pillow_Object',
 'pillows_Object',
 'ping_pong_paddle_Object',
 'ping_pong_table_Object',
 'pipe_Object',
 'pipes_Object',
 'pitcher_Object',
 'pizza_box_Object',
 'pizza_boxes_Object',
 'plant_Object',
 'plastic_bin_Object',
 'plastic_container_Object',
 'plastic_containers_Object',
 'plastic_storage_bin_Object',
 'plate_Object',
 'plates_Object',
 'platform_Object',
 'plunger_Object',
 'podium_Object',
 'pool_table_Object',
 'postcard_Object',
 'poster_Object',
 'poster_cutter_Object',
 'poster_printer_Object',
 'poster_tube_Object',
 'pot_Object',
 'potted_plant_Object',
 'power_outlet_Object',
 'power_strip_Object',
 'printer_Object',
 'projector_Object',
 'projector_screen_Object',
 'purse_Object',
 'quadcopter_Object',
 'rack_Object',
 'rack_stand_Object',
 'radiator_Object',
 'rail_Object',
 'railing_Object',
 'range_hood_Object',
 'recliner_chair_Object',
 'recycling_bin_Object',
 'refrigerator_Object',
 'remote_Object',
 'rice_cooker_Object',
 'rocking_chair_Object',
 'rod_Object',
 'rolled_poster_Object',
 'roomba_Object',
 'rope_Object',
 'round_table_Object',
 'rug_Object',
 'salt_Object',
 'santa_Object',
 'scale_Object',
 'scanner_Object',
 'screen_Object',
 'seat_Object',
 'seating_Object',
 'sewing_machine_Object',
 'shampoo_Object',
 'shampoo_bottle_Object',
 'shaving_cream_Object',
 'shelf_Object',
 'shirt_Object',
 'shoe_Object',
 'shoe_rack_Object',
 'shoes_Object',
 'shopping_bag_Object',
 'shorts_Object',
 'shower_Object',
 'shower_control_valve_Object',
 'shower_curtain_Object',
 'shower_curtain_rod_Object',
 'shower_door_Object',
 'shower_doors_Object',
 'shower_faucet_handle_Object',
 'shower_floor_Object',
 'shower_head_Object',
 'shower_wall_Object',
 'shower_walls_Object',
 'shredder_Object',
 'side_table_Object',
 'sign_Object',
 'sink_Object',
 'sleeping_bag_Object',
 'sliding_door_Object',
 'sliding_wood_door_Object',
 'slipper_Object',
 'slippers_Object',
 'smoke_detector_Object',
 'soap_Object',
 'soap_bar_Object',
 'soap_bottle_Object',
 'soap_dish_Object',
 'soap_dispenser_Object',
 'sock_Object',
 'socks_Object',
 'soda_can_Object',
 'soda_stream_Object',
 'sofa_Object',
 'sofa_bed_Object',
 'sofa_chair_Object',
 'speaker_Object',
 'sponge_Object',
 'spray_bottle_Object',
 'stack_of_chairs_Object',
 'stack_of_cups_Object',
 'stack_of_folded_chairs_Object',
 'stacks_of_cups_Object',
 'stair_Object',
 'stair_rail_Object',
 'staircase_Object',
 'stairs_Object',
 'stand_Object',
 'stapler_Object',
 'star_Object',
 'starbucks_cup_Object',
 'statue_Object',
 'step_Object',
 'step_stool_Object',
 'stepladder_Object',
 'stepstool_Object',
 'stick_Object',
 'sticker_Object',
 'stool_Object',
 'stools_Object',
 'storage_bin_Object',
 'storage_box_Object',
 'storage_container_Object',
 'storage_organizer_Object',
 'storage_shelf_Object',
 'stove_Object',
 'stovetop_Object',
 'structure_Object',
 'studio_light_Object',
 'stuffed_animal_Object',
 'subwoofer_Object',
 'suitcase_Object',
 'suitcases_Object',
 'sweater_Object',
 'swiffer_Object',
 'switch_Object',
 'table_Object',
 'table_lamp_Object',
 'tank_Object',
 'tap_Object',
 'tape_Object',
 'tea_kettle_Object',
 'teapot_Object',
 'teddy_bear_Object',
 'telephone_Object',
 'telescope_Object',
 'tennis_racket_Object',
 'tennis_racket_bag_Object',
 'thermos_Object',
 'thermostat_Object',
 'tire_Object',
 'tissue_box_Object',
 'toaster_Object',
 'toaster_oven_Object',
 'toilet_Object',
 'toilet_brush_Object',
 'toilet_flush_button_Object',
 'toilet_paper_Object',
 'toilet_paper_dispenser_Object',
 'toilet_paper_holder_Object',
 'toilet_paper_package_Object',
 'toilet_paper_rolls_Object',
 'toilet_seat_cover_dispenser_Object',
 'toiletry_Object',
 'toolbox_Object',
 'toothbrush_Object',
 'toothpaste_Object',
 'towel_Object',
 'towel_rack_Object',
 'towels_Object',
 'toy_dinosaur_Object',
 'toy_piano_Object',
 'traffic_cone_Object',
 'trash_bag_Object',
 'trash_bin_Object',
 'trash_cabinet_Object',
 'trash_can_Object',
 'tray_Object',
 'tray_rack_Object',
 'treadmill_Object',
 'tripod_Object',
 'trolley_Object',
 'trunk_Object',
 'tube_Object',
 'tupperware_Object',
 'tv_Object',
 'tv_stand_Object',
 'umbrella_Object',
 'urinal_Object',
 'vacuum_cleaner_Object',
 'vase_Object',
 'vending_machine_Object',
 'vent_Object',
 'wall_Object',
 'wall_hanging_Object',
 'wall_lamp_Object',
 'wall_mounted_coat_rack_Object',
 'wardrobe_Object',
 'wardrobe_cabinet_Object',
 'wardrobe_closet_Object',
 'washcloth_Object',
 'washing_machine_Object',
 'washing_machines_Object',
 'water_bottle_Object',
 'water_cooler_Object',
 'water_fountain_Object',
 'water_heater_Object',
 'water_pitcher_Object',
 'water_softener_Object',
 'wet_floor_sign_Object',
 'wheel_Object',
 'whiteboard_Object',
 'whiteboard_eraser_Object',
 'wig_Object',
 'window_Object',
 'windowsill_Object',
 'wood_Object',
 'wood_beam_Object',
 'workbench_Object',
 'xbox_controller_Object',
 'yoga_mat_Object']

view_concepts = ['left_Object_Object',
 'right_Object_Object',
 'back_Object_Object',
 'behind_Object_Object']
