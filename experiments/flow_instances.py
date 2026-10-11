from briskit.flow import (DAMAGED_COUNTS, DATA_DIR, FAMILIES, SEEDS, generate_instance, handmade_instance,
                          save_instance)


def main():
    save_instance(handmade_instance(), DATA_DIR / "handmade.json")
    for damaged in DAMAGED_COUNTS:
        for family in FAMILIES:
            for seed in SEEDS:
                instance = generate_instance(family, damaged, seed)
                save_instance(instance, DATA_DIR / f"{instance['name']}.json")
                print(f"   {instance['name']:24s} {len(instance['network']['links']):3d} links,"
                      f" {len(instance['elements'])} damaged, {len(instance['demands'])} commodities")


if __name__ == "__main__":
    main()
